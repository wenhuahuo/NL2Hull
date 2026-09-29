#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=v043-kev-jev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --chdir=/data/home/wenhua_huo/Documents/NurbsShipGen
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
MODE="${1:?pass smoke or full}"
PROJECT_DIR="/data/home/wenhua_huo/Documents/NurbsShipGen"
DATA_FILE="${PROJECT_DIR}/datasets/kev_ship_dataset/test_5k.jsonl"
DATA_MANIFEST="${PROJECT_DIR}/datasets/kev_ship_dataset/test_5k_manifest.json"
KEV_SOURCE="${PROJECT_DIR}/third_party/kev"
EVAL_SCRIPT="${PROJECT_DIR}/scripts/evaluate/run_kev_jev_eval.py"
case "${MODE}" in
  smoke) LIMIT=2; OUTPUT_DIR="${PROJECT_DIR}/outputs/v043_kev_jev_openrouter_smoke_v2" ;;
  full) LIMIT=5000; OUTPUT_DIR="${PROJECT_DIR}/outputs/v043_kev_jev_openrouter_test_5k_v2" ;;
  *) echo "ERROR: choose smoke or full" >&2; exit 2 ;;
esac

if ! command -v module >/dev/null 2>&1; then source /etc/profile.d/openscow-modules.sh; fi
module purge
module load miniconda3/py313_26.3.2-2
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
source "${HOME}/.conda/envs/kev-v039/bin/activate"
cd "${PROJECT_DIR}"
[[ -r "${DATA_FILE}" && -r "${DATA_MANIFEST}" && -r "${EVAL_SCRIPT}" && -r .env ]] || {
  echo "ERROR: missing frozen dataset, evaluator, or .env" >&2; exit 2
}
[[ -d "${KEV_SOURCE}/kev" && ! -e "${OUTPUT_DIR}" ]] || {
  echo "ERROR: missing Kev source or refusing to overwrite ${OUTPUT_DIR}" >&2; exit 2
}
python - "${DATA_FILE}" "${DATA_MANIFEST}" <<'PY'
import hashlib, json, sys
from pathlib import Path
p, m = map(Path, sys.argv[1:])
manifest = json.loads(m.read_text())
if manifest['subset_records'] != 5000 or hashlib.sha256(p.read_bytes()).hexdigest() != manifest['subset_sha256']:
    raise SystemExit('frozen subset mismatch')
print(f"verified frozen 5k dataset: {manifest['subset_sha256']}")
PY
set -a
source .env
set +a
[[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "ERROR: OPENROUTER_API_KEY is not configured" >&2; exit 2; }
export PYTHONPATH="${PROJECT_DIR}/src:${KEV_SOURCE}"
printf 'job=%s mode=%s model=%s records=%s start=%s\n' \
  "${SLURM_JOB_ID}" "${MODE}" '~typesafe/jev-latest' "${LIMIT}" "$(date -Is)"
python -u "${EVAL_SCRIPT}" --data "${DATA_FILE}" --output "${OUTPUT_DIR}" \
  --limit "${LIMIT}" --timeout 180
printf 'job=%s finished=%s output=%s\n' "${SLURM_JOB_ID}" "$(date -Is)" "${OUTPUT_DIR}"
