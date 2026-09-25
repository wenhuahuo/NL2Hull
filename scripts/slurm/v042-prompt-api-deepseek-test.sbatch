#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=v042-prompt-api-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --chdir=/data/home/wenhua_huo/Documents/NurbsShipGen
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1
PROJECT_DIR="/data/home/wenhua_huo/Documents/NurbsShipGen"
DATA_FILE="${PROJECT_DIR}/outputs/v042_prompt_eval_dataset/test.jsonl"
OUTPUT_DIR="${PROJECT_DIR}/outputs/v042_prompt_api_deepseek_smoke"

if ! command -v module >/dev/null 2>&1; then source /etc/profile.d/openscow-modules.sh; fi
module purge
module load miniconda3/py313_26.3.2-2
module load pi/0.87.0
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
cd "${PROJECT_DIR}"
[[ -r "${DATA_FILE}" && -r .env ]] || { echo "ERROR: missing API evaluation input or .env" >&2; exit 2; }
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "ERROR: refusing to overwrite ${OUTPUT_DIR}" >&2; exit 2; }
command -v pi >/dev/null || { echo "ERROR: pi is unavailable" >&2; exit 2; }
set -a
source .env
set +a
export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}/scripts/jev"

printf 'job=%s host=%s model=%s thinking=%s records=%s start=%s\n' \
  "${SLURM_JOB_ID}" "$(hostname)" "deepseek/deepseek-flash" "low" "100" "$(date -Is)"
python -u scripts/jev/run_prompt_action_api_eval.py \
  --data "${DATA_FILE}" \
  --output "${OUTPUT_DIR}" \
  --model "deepseek/deepseek-flash" \
  --thinking low \
  --limit 100 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --timeout 180
printf 'job=%s finished=%s output=%s\n' "${SLURM_JOB_ID}" "$(date -Is)" "${OUTPUT_DIR}"
