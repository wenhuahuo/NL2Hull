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
LIMIT="${1:-}"
PROJECT_DIR=/data/home/wenhua_huo/Documents/NurbsShipGen
KEV_ENV="${HOME}/.conda/envs/kev-v039"
DATA_FILE="${PROJECT_DIR}/datasets/kev_ship_dataset/test_5k.jsonl"
DATA_MANIFEST="${PROJECT_DIR}/datasets/kev_ship_dataset/test_5k_manifest.json"
EVAL_SCRIPT="${PROJECT_DIR}/scripts/jev/run_prompt_action_transformers_eval.py"
MODEL_DIR="${PROJECT_DIR}/third_party/models/Qwen3.5-4B"
if [[ -n "${LIMIT}" ]]; then
  OUTPUT_DIR="${PROJECT_DIR}/outputs/v046_qwen35_test1_smoke"
else
  OUTPUT_DIR="${PROJECT_DIR}/outputs/v046_qwen35_test5k_eval"
fi

if ! command -v module >/dev/null 2>&1; then source /etc/profile.d/openscow-modules.sh; fi
module purge
module load miniconda3/py313_26.3.2-2
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
source "${KEV_ENV}/bin/activate"
cd "${PROJECT_DIR}"
[[ -r "${DATA_FILE}" && -r "${DATA_MANIFEST}" && -r "${EVAL_SCRIPT}" ]] || { echo "ERROR: missing input or script" >&2; exit 2; }
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
export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}/scripts/jev"
export HF_HOME="${PROJECT_DIR}/third_party/hf-cache"
export HF_HUB_OFFLINE=1
printf 'job=%s model=qwen35-transformers-direct records=%s data_sha256=%s start=%s\n' \
  "${SLURM_JOB_ID}" "$(wc -l < "${DATA_FILE}")" \
  "$(sha256sum "${DATA_FILE}" | cut -d" " -f1)" "$(date -Is)"
EVAL_ARGS=(
  --data "${DATA_FILE}"
  --model "${MODEL_DIR}"
  --output "${OUTPUT_DIR}"
)
if [[ -n "${LIMIT}" ]]; then EVAL_ARGS+=(--limit "${LIMIT}"); fi
python -u "${EVAL_SCRIPT}" "${EVAL_ARGS[@]}"
printf 'job=%s finished=%s output=%s\n' "${SLURM_JOB_ID}" "$(date -Is)" "${OUTPUT_DIR}"
