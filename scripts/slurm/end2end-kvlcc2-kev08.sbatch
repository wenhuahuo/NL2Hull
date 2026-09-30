#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --job-name=end2end-kvlcc2-kev08
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/data/home/wenhua_huo/Documents/NurbsShipGen
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
PROJECT_DIR=/data/home/wenhua_huo/Documents/NurbsShipGen
KEV_SOURCE="${PROJECT_DIR}/third_party/kev"
KEV_ENV="${HOME}/.conda/envs/kev-v039"
MESH_ENV="${HOME}/.conda/envs/mesh"
DATA_FILE="${PROJECT_DIR}/datasets/kev_ship_dataset/train.jsonl"
RUN_DIR="${PROJECT_DIR}/outputs/v042_kev0.8b_ship"
HULL_STATE="${PROJECT_DIR}/datasets/v036_structured_action_dataset_30000/hull_states/KVLCC2.json"
OUTPUT_DIR="${PROJECT_DIR}/outputs/end2end_kvlcc2_kev08"
EVAL_SCRIPT="${PROJECT_DIR}/scripts/evaluate/run_end2end_eval.py"
PER_CATEGORY="${1:-100}"

if ! command -v module >/dev/null 2>&1; then source /etc/profile.d/openscow-modules.sh; fi
module purge
module load miniconda3/py313_26.3.2-2
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
cd "${PROJECT_DIR}"
export HF_HOME="${PROJECT_DIR}/third_party/hf-cache"
export HF_HUB_OFFLINE=1

[[ -r "${DATA_FILE}" && -r "${HULL_STATE}" && -r "${EVAL_SCRIPT}" ]] || {
  echo "ERROR: missing evaluation input or script" >&2; exit 2
}
[[ -f "${RUN_DIR}/adapter_model.safetensors" && -f "${RUN_DIR}/head.pt" && -f "${RUN_DIR}/training_config.json" ]] || {
  echo "ERROR: incomplete Kev checkpoint ${RUN_DIR}" >&2; exit 2
}
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "ERROR: refusing to overwrite ${OUTPUT_DIR}" >&2; exit 2; }

printf 'job=%s phase=predict per_category=%s start=%s\n' "${SLURM_JOB_ID}" "${PER_CATEGORY}" "$(date -Is)"
source "${KEV_ENV}/bin/activate"
PYTHONPATH="${PROJECT_DIR}/src:${KEV_SOURCE}" python -u "${EVAL_SCRIPT}" predict \
  --data "${DATA_FILE}" \
  --run "${RUN_DIR}" \
  --output "${OUTPUT_DIR}" \
  --hull KVLCC2 \
  --per-category "${PER_CATEGORY}" \
  --device cuda
conda deactivate

printf 'job=%s phase=execute start=%s\n' "${SLURM_JOB_ID}" "$(date -Is)"
source "${MESH_ENV}/bin/activate"
PYTHONPATH="${PROJECT_DIR}/src" python -u "${EVAL_SCRIPT}" execute \
  --output "${OUTPUT_DIR}" \
  --hull-state "${HULL_STATE}"

printf 'job=%s phase=visualize start=%s\n' "${SLURM_JOB_ID}" "$(date -Is)"
PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/src" python -u -m visualization.end2end_test_vis.plot_end2end \
  --results "${OUTPUT_DIR}"
conda deactivate

printf 'job=%s finished=%s output=%s\n' "${SLURM_JOB_ID}" "$(date -Is)" "${OUTPUT_DIR}"
