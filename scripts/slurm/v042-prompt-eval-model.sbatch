#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --job-name=v042-prompt-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/data/home/wenhua_huo/Documents/NurbsShipGen
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

MODEL_KEY="${1:?model key required: qwen35, llama32, or gemma3}"
PROJECT_DIR="/data/home/wenhua_huo/Documents/NurbsShipGen"
DATA_DIR="${PROJECT_DIR}/outputs/v042_prompt_eval_dataset"
DATA_FILE="${DATA_DIR}/test.jsonl"
EVAL_SCRIPT="${PROJECT_DIR}/scripts/jev/run_prompt_action_eval.py"
KEV_ENV="${HOME}/.conda/envs/kev-v039"
VLLM_ENV="${HOME}/.conda/envs/vllm-qwen3"
QWEN_DIR="${PROJECT_DIR}/third_party/models/Qwen3.5-4B"
LLAMA_DIR="${PROJECT_DIR}/third_party/models/llama-3.2-3b-instruct"
GEMMA_DIR="${PROJECT_DIR}/third_party/models/gemma-3-4b-it"
PORT=8000
BASE_URL="http://127.0.0.1:${PORT}/v1"
OUTPUT_DIR="${PROJECT_DIR}/outputs/v042_prompt_eval_${MODEL_KEY}"
SERVER_LOG="${PROJECT_DIR}/logs/apps/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-server.log"

if ! command -v module >/dev/null 2>&1; then source /etc/profile.d/openscow-modules.sh; fi
module purge
module load gcc/13.3.0
module load miniconda3/py313_26.3.2-2
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

case "${MODEL_KEY}" in
  qwen35)
    MODEL_DIR="${QWEN_DIR}"
    SERVER_KIND="transformers"
    SERVE_MODEL="${MODEL_DIR}"
    ;;
  llama32)
    MODEL_DIR="${LLAMA_DIR}"
    SERVER_KIND="vllm"
    SERVE_MODEL="${MODEL_DIR}"
    SERVED_NAME="llama-3.2-3b-instruct"
    ;;
  gemma3)
    MODEL_DIR="${GEMMA_DIR}"
    SERVER_KIND="vllm"
    SERVE_MODEL="${MODEL_DIR}"
    SERVED_NAME="gemma-3-4b-it"
    ;;
  *) echo "ERROR: unknown model key ${MODEL_KEY}" >&2; exit 2 ;;
esac

[[ -r "${DATA_FILE}" && -r "${EVAL_SCRIPT}" ]] || { echo "ERROR: missing evaluation input or script" >&2; exit 2; }
[[ -f "${MODEL_DIR}/config.json" ]] || { echo "ERROR: missing model ${MODEL_DIR}" >&2; exit 2; }
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "ERROR: refusing to overwrite ${OUTPUT_DIR}" >&2; exit 2; }
mkdir -p logs/apps

source "${KEV_ENV}/bin/activate"
export PYTHONPATH="${PROJECT_DIR}/src"
export HF_HOME="${PROJECT_DIR}/third_party/hf-cache"
export HF_ENDPOINT="https://hf-mirror.com"

SERVER_PID=""
cleanup() { [[ -z "${SERVER_PID}" ]] || kill "${SERVER_PID}" >/dev/null 2>&1 || true; }
trap cleanup EXIT

if [[ "${SERVER_KIND}" == "transformers" ]]; then
  "${KEV_ENV}/bin/transformers" serve "${SERVE_MODEL}" \
    --host 127.0.0.1 --port "${PORT}" --device cuda:0 --dtype bfloat16 \
    --reasoning off --no-continuous-batching --log-level warning \
    >"${SERVER_LOG}" 2>&1 &
else
  "${VLLM_ENV}/bin/vllm" serve "${SERVE_MODEL}" \
    --served-model-name "${SERVED_NAME}" --host 127.0.0.1 --port "${PORT}" \
    --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.85 \
    >"${SERVER_LOG}" 2>&1 &
fi
SERVER_PID=$!

MODEL_ID=""
for _ in $(seq 1 180); do
  if curl -sf "${BASE_URL}/models" >"/tmp/${SLURM_JOB_ID}-models.json"; then
    MODEL_ID="$(${KEV_ENV}/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["data"][0]["id"])' "/tmp/${SLURM_JOB_ID}-models.json")"
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "ERROR: model server exited during startup" >&2
    tail -80 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  sleep 2
done
[[ -n "${MODEL_ID}" ]] || { echo "ERROR: model server did not become ready" >&2; tail -80 "${SERVER_LOG}" >&2 || true; exit 1; }

printf 'job=%s model_key=%s model_id=%s server=%s host=%s gpu=%s start=%s\n' \
  "${SLURM_JOB_ID}" "${MODEL_KEY}" "${MODEL_ID}" "${SERVER_KIND}" "$(hostname)" \
  "${CUDA_VISIBLE_DEVICES:-unset}" "$(date -Is)"
"${KEV_ENV}/bin/python" -u "${EVAL_SCRIPT}" \
  --data "${DATA_FILE}" --output "${OUTPUT_DIR}" --base-url "${BASE_URL}" \
  --model "${MODEL_ID}" --workers 4 --timeout 180
printf 'job=%s finished=%s output=%s\n' "${SLURM_JOB_ID}" "$(date -Is)" "${OUTPUT_DIR}"
