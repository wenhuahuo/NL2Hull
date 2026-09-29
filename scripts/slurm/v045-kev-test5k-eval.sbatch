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
PROJECT_DIR=/data/home/wenhua_huo/Documents/NurbsShipGen
KEV_SOURCE="${PROJECT_DIR}/third_party/kev"
KEV_ENV="${HOME}/.conda/envs/kev-v039"
DATA_FILE="${PROJECT_DIR}/datasets/kev_ship_dataset/test_5k.jsonl"
DATA_MANIFEST="${PROJECT_DIR}/datasets/kev_ship_dataset/test_5k_manifest.json"
EVAL_SCRIPT="${PROJECT_DIR}/scripts/evaluate/run_kev_eval.py"

case "${MODEL_KEY}" in
  kev08)
    RUN_DIR="${PROJECT_DIR}/outputs/v042_kev0.8b_ship"
    OUTPUT_DIR="${PROJECT_DIR}/outputs/v045_kev08_test5k_eval"
    ;;
  kev2b)
    RUN_DIR="${PROJECT_DIR}/outputs/v042_kev2b_ship"
    OUTPUT_DIR="${PROJECT_DIR}/outputs/v045_kev2b_test5k_eval"
    ;;
  kev4b)
    RUN_DIR="${PROJECT_DIR}/outputs/v041_kev4b_ship_cleaned/checkpoint"
    OUTPUT_DIR="${PROJECT_DIR}/outputs/v045_kev4b_test5k_eval"
    ;;
  *) echo "ERROR: unknown model key ${MODEL_KEY}" >&2; exit 2 ;;
esac

if ! command -v module >/dev/null 2>&1; then source /etc/profile.d/openscow-modules.sh; fi
module purge
module load miniconda3/py313_26.3.2-2
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
source "${KEV_ENV}/bin/activate"
cd "${PROJECT_DIR}"

[[ -r "${DATA_FILE}" && -r "${DATA_MANIFEST}" && -r "${EVAL_SCRIPT}" ]] || {
  echo "ERROR: missing evaluation input or script" >&2; exit 2
}
[[ -f "${RUN_DIR}/adapter_model.safetensors" && -f "${RUN_DIR}/head.pt" && -f "${RUN_DIR}/training_config.json" ]] || {
  echo "ERROR: incomplete Kev checkpoint ${RUN_DIR}" >&2; exit 2
}
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
export PYTHONPATH="${PROJECT_DIR}/src:${KEV_SOURCE}"
export HF_HOME="${PROJECT_DIR}/third_party/hf-cache"
export HF_HUB_OFFLINE=1
printf 'job=%s model=%s run=%s records=%s data_sha256=%s start=%s\n' \
  "${SLURM_JOB_ID}" "${MODEL_KEY}" "${RUN_DIR}" "$(wc -l < "${DATA_FILE}")" \
  "$(sha256sum "${DATA_FILE}" | cut -d" " -f1)" "$(date -Is)"
python -u "${EVAL_SCRIPT}" \
  --run "${RUN_DIR}" \
  --data "${DATA_FILE}" \
  --output "${OUTPUT_DIR}" \
  --device cuda
printf 'job=%s finished=%s output=%s\n' "${SLURM_JOB_ID}" "$(date -Is)" "${OUTPUT_DIR}"
