#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --job-name=nurbs-v038-full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=8G
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/data/home/wenhua_huo/Documents/NurbsShipGen
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/openscow-modules.sh
fi
module purge
module load miniconda3/py313_26.3.2-2
module load pi/0.87.0
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-/data/home/wenhua_huo/Documents/NurbsShipGen}"
cd "${PROJECT_DIR}"
set -a
source .env
set +a
export PYTHONPATH="${PROJECT_DIR}/src"
CODE_REVISION_FILE="${PROJECT_DIR}/scripts/slurm/v038-code-revision.txt"
if [[ -r "${CODE_REVISION_FILE}" ]]; then
  NURBS_CODE_COMMIT="$(cat "${CODE_REVISION_FILE}")"
fi
export NURBS_CODE_COMMIT="${NURBS_CODE_COMMIT:-unknown}"

RUNTIME_DIR="${PROJECT_DIR}/.pi-runtime-v038"
mkdir -p "${RUNTIME_DIR}"
cp "$HOME/.pi/agent/auth.json" "${RUNTIME_DIR}/auth.json"
cp "$HOME/.pi/agent/models-store.json" "${RUNTIME_DIR}/models-store.json"
cp "${PROJECT_DIR}/scripts/slurm/v038-models.json" "${RUNTIME_DIR}/models.json"
python - <<'PY'
import json
from pathlib import Path
path = Path(".pi-runtime-v038/auth.json")
data = json.loads(path.read_text())
data.pop("deepseek", None)
path.write_text(json.dumps(data))
PY
export PI_CODING_AGENT_DIR="${RUNTIME_DIR}"

mkdir -p logs/apps outputs
printf 'job=%s host=%s start=%s project=%s commit=%s\n' \
  "${SLURM_JOB_ID}" "$(hostname)" "$(date -Is)" "${PROJECT_DIR}" "${NURBS_CODE_COMMIT}"
python --version
pi --version

OUTPUT_DIR="outputs/v038_stage3_language_150k_full"
RESUME_ARGS=()
if [[ -e "${OUTPUT_DIR}" ]]; then
  RESUME_ARGS=(--resume)
fi
python -u -m nurbs_ship_reconstruction.language.generate_descriptions \
  --source outputs/v036_structured_action_dataset_30000 \
  --output "${OUTPUT_DIR}" \
  --workers 7 --progress-every 10 --timeout 180 \
  "${RESUME_ARGS[@]}"

printf 'job=%s finished=%s\n' "${SLURM_JOB_ID}" "$(date -Is)"
