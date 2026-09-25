#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=v042-kev-api-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --chdir=/data/home/wenhua_huo/Documents/NurbsShipGen
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1
PROJECT_DIR="/data/home/wenhua_huo/Documents/NurbsShipGen"
KEV_SOURCE="${PROJECT_DIR}/third_party/kev"
DATA_FILE="${PROJECT_DIR}/datasets/kev_ship_dataset/test.jsonl"
EVAL_SCRIPT="${PROJECT_DIR}/scripts/jev/run_kev_pi_eval.py"
OUTPUT_DIR="${PROJECT_DIR}/outputs/v042_kev_api_deepseek_test"

if ! command -v module >/dev/null 2>&1; then source /etc/profile.d/openscow-modules.sh; fi
module purge
module load miniconda3/py313_26.3.2-2
module load pi/0.87.0
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
cd "${PROJECT_DIR}"
[[ -r "${DATA_FILE}" && -r "${EVAL_SCRIPT}" && -r .env ]] || { echo "ERROR: missing Kev API evaluation input, script, or .env" >&2; exit 2; }
[[ -d "${KEV_SOURCE}/kev" ]] || { echo "ERROR: missing Kev source" >&2; exit 2; }
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "ERROR: refusing to overwrite ${OUTPUT_DIR}" >&2; exit 2; }
command -v pi >/dev/null || { echo "ERROR: pi is unavailable" >&2; exit 2; }
set -a
source .env
set +a
export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}/scripts/jev:${KEV_SOURCE}"

printf 'job=%s host=%s model=%s thinking=%s records=%s data_sha256=%s start=%s\n' \
  "${SLURM_JOB_ID}" "$(hostname)" "deepseek/deepseek-flash" "low" \
  "$(wc -l < "${DATA_FILE}")" "$(sha256sum "${DATA_FILE}" | cut -d' ' -f1)" "$(date -Is)"
python -u "${EVAL_SCRIPT}" \
  --data "${DATA_FILE}" \
  --output "${OUTPUT_DIR}" \
  --model "deepseek/deepseek-flash" \
  --thinking low \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --timeout 180 \
  --retries 2
printf 'job=%s finished=%s output=%s\n' "${SLURM_JOB_ID}" "$(date -Is)" "${OUTPUT_DIR}"
