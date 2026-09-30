#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --chdir=/data/home/wenhua_huo/Documents/NurbsShipGen
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

MODEL_KEY="${1:?model key required: kev08, kev2b, or kev4b}"
TIER="${2:?tier required: original, easy, or hard}"
PROJECT_DIR="/data/home/wenhua_huo/Documents/NurbsShipGen"
KEV_SOURCE="${PROJECT_DIR}/third_party/kev"
KEV_ENV="${HOME}/.conda/envs/kev-v039"
DATA_DIR="${PROJECT_DIR}/datasets/jevbench"
EVAL_SCRIPT="${PROJECT_DIR}/scripts/evaluate/run_jevbench_kev_eval.py"

case "${MODEL_KEY}" in
  kev08)
    RUN_DIR="${PROJECT_DIR}/outputs/v042_kev0.8b_ship"
    EXPERIMENT="v046_kev08_jevbench_${TIER}"
    ;;
  kev2b)
    RUN_DIR="${PROJECT_DIR}/outputs/v042_kev2b_ship"
    EXPERIMENT="v046_kev2b_jevbench_${TIER}"
    ;;
  kev4b)
    RUN_DIR="${PROJECT_DIR}/outputs/v041_kev4b_ship_cleaned/checkpoint"
    EXPERIMENT="v046_kev4b_jevbench_${TIER}"
    ;;
  *) echo "ERROR: unknown model key ${MODEL_KEY}" >&2; exit 2 ;;
esac
case "${TIER}" in
  original|easy|hard) DATA_FILE="${DATA_DIR}/${TIER}.jsonl" ;;
  *) echo "ERROR: unknown JevBench tier ${TIER}" >&2; exit 2 ;;
esac
OUTPUT_DIR="${PROJECT_DIR}/outputs/${EXPERIMENT}"

if ! command -v module >/dev/null 2>&1; then source /etc/profile.d/openscow-modules.sh; fi
module purge
module load miniconda3/py313_26.3.2-2
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
source "${KEV_ENV}/bin/activate"
cd "${PROJECT_DIR}"

[[ -r "${DATA_FILE}" && -r "${EVAL_SCRIPT}" ]] || { echo "ERROR: missing JevBench input or evaluator" >&2; exit 2; }
[[ -d "${KEV_SOURCE}/kev" ]] || { echo "ERROR: missing Kev source" >&2; exit 2; }
[[ -f "${RUN_DIR}/adapter_model.safetensors" && -f "${RUN_DIR}/head.pt" && -f "${RUN_DIR}/training_config.json" ]] || {
  echo "ERROR: incomplete Kev checkpoint ${RUN_DIR}" >&2; exit 2;
}
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "ERROR: refusing to overwrite ${OUTPUT_DIR}" >&2; exit 2; }

export PYTHONPATH="${PROJECT_DIR}/src:${KEV_SOURCE}"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="${PROJECT_DIR}/third_party/hf-cache"
printf 'job=%s model=%s tier=%s run=%s records=%s data_sha256=%s hf_endpoint=%s start=%s\n' \
  "${SLURM_JOB_ID}" "${MODEL_KEY}" "${TIER}" "${RUN_DIR}" "$(wc -l < "${DATA_FILE}")" \
  "$(sha256sum "${DATA_FILE}" | cut -d" " -f1)" "${HF_ENDPOINT}" "$(date -Is)"
python -u "${EVAL_SCRIPT}" \
  --run "${RUN_DIR}" \
  --data "${DATA_FILE}" \
  --output "${OUTPUT_DIR}" \
  --device cuda
printf 'job=%s finished=%s output=%s\n' "${SLURM_JOB_ID}" "$(date -Is)" "${OUTPUT_DIR}"
