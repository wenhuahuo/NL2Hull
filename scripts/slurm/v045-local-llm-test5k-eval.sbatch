#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --chdir=/data/home/wenhua_huo/Documents/NurbsShipGen
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
MODEL_KEY="${1:?model key required: llama32, qwen35, or gemma3}"
PROJECT_DIR=/data/home/wenhua_huo/Documents/NurbsShipGen
DATA_FILE="${PROJECT_DIR}/datasets/kev_ship_dataset/test_5k.jsonl"
DATA_MANIFEST="${PROJECT_DIR}/datasets/kev_ship_dataset/test_5k_manifest.json"
EVAL_SCRIPT="${PROJECT_DIR}/scripts/jev/run_prompt_action_eval.py"
KEV_ENV="${HOME}/.conda/envs/kev-v039"
VLLM_ENV="${HOME}/.conda/envs/vllm-qwen3"
QWEN_DIR="${PROJECT_DIR}/third_party/models/Qwen3.5-4B"
LLAMA_DIR="${PROJECT_DIR}/third_party/models/llama-3.2-3b-instruct"
GEMMA_DIR="${PROJECT_DIR}/third_party/models/gemma-3-4b-it"
PORT=8000
BASE_URL="http://127.0.0.1:${PORT}/v1"

case "${MODEL_KEY}" in
  qwen35)
    MODEL_DIR="${QWEN_DIR}"
    SERVER_KIND=transformers
    OUTPUT_DIR="${PROJECT_DIR}/outputs/v045_qwen35_test5k_eval"
    ;;
  llama32)
    MODEL_DIR="${LLAMA_DIR}"
    SERVER_KIND=vllm
    SERVED_NAME=llama-3.2-3b-instruct
    OUTPUT_DIR="${PROJECT_DIR}/outputs/v045_llama32_test5k_eval"
    ;;
  gemma3)
    MODEL_DIR="${GEMMA_DIR}"
    SERVER_KIND=vllm
    SERVED_NAME=gemma-3-4b-it
    OUTPUT_DIR="${PROJECT_DIR}/outputs/v045_gemma3_test5k_eval"
    ;;
  *) echo "ERROR: unknown model key ${MODEL_KEY}" >&2; exit 2 ;;
esac

if ! command -v module >/dev/null 2>&1; then source /etc/profile.d/openscow-modules.sh; fi
module purge
module load gcc/13.3.0
module load miniconda3/py313_26.3.2-2
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
source "${KEV_ENV}/bin/activate"
cd "${PROJECT_DIR}"

[[ -r "${DATA_FILE}" && -r "${DATA_MANIFEST}" && -r "${EVAL_SCRIPT}" ]] || {
  echo "ERROR: missing evaluation input or script" >&2; exit 2
}
[[ -f "${MODEL_DIR}/config.json" ]] || { echo "ERROR: missing model ${MODEL_DIR}" >&2; exit 2; }
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "ERROR: refusing to overwrite ${OUTPUT_DIR}" >&2; exit 2; }
python - "${DATA_FILE}" "${DATA_MANIFEST}" <<'PY'
import hashlib, json, sys
from pathlib import Path
path, manifest_path = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text())
hash_value = hashlib.sha256(path.read_bytes()).hexdigest()
if manifest.get("subset_records") != 5000 or manifest.get("subset_sha256") != hash_value:
    raise SystemExit("frozen test_5k subset mismatch")
print(f"verified records=5000 sha256={hash_value}")
PY

SERVER_LOG="${PROJECT_DIR}/logs/apps/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-server.log"
mkdir -p "${PROJECT_DIR}/logs/apps"
SERVER_PID=""
cleanup() { [[ -z "${SERVER_PID}" ]] || kill "${SERVER_PID}" >/dev/null 2>&1 || true; }
trap cleanup EXIT
export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}/scripts/jev"
export HF_HOME="${PROJECT_DIR}/third_party/hf-cache"
export HF_ENDPOINT="https://hf-mirror.com"

if [[ "${SERVER_KIND}" == "transformers" ]]; then
  "${KEV_ENV}/bin/transformers" serve "${MODEL_DIR}" \
    --host 127.0.0.1 --port "${PORT}" --device cuda:0 --dtype bfloat16 \
    --reasoning off --no-continuous-batching --log-level warning \
    >"${SERVER_LOG}" 2>&1 &
else
  [[ -x "${VLLM_ENV}/bin/vllm" ]] || { echo "ERROR: vLLM is unavailable" >&2; exit 2; }
  "${VLLM_ENV}/bin/vllm" serve "${MODEL_DIR}" \
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
    tail -n 80 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  sleep 2
done
[[ -n "${MODEL_ID}" ]] || { echo "ERROR: model server did not become ready" >&2; tail -n 80 "${SERVER_LOG}" >&2 || true; exit 1; }

printf 'job=%s model=%s model_id=%s server=%s records=%s data_sha256=%s start=%s\n' \
  "${SLURM_JOB_ID}" "${MODEL_KEY}" "${MODEL_ID}" "${SERVER_KIND}" \
  "$(wc -l < "${DATA_FILE}")" "$(sha256sum "${DATA_FILE}" | cut -d" " -f1)" "$(date -Is)"
"${KEV_ENV}/bin/python" -u "${EVAL_SCRIPT}" \
  --data "${DATA_FILE}" \
  --output "${OUTPUT_DIR}" \
  --base-url "${BASE_URL}" \
  --model "${MODEL_ID}" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --timeout 180
printf 'job=%s finished=%s output=%s\n' "${SLURM_JOB_ID}" "$(date -Is)" "${OUTPUT_DIR}"
