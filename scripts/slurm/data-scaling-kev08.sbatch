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

# Data-scaling ablation for Kev-0.8B: nested training subsets evaluated on the
# frozen test_5k benchmark. Lane A trains the 25% subset; lane B evaluates the
# published Kev-0.8B (0% point) then trains the 1/2/4/8% subsets in order.

set -euo pipefail
LANE="${1:?lane required: lanea or laneb}"
PROJECT_DIR=/data/home/wenhua_huo/Documents/NurbsShipGen
KEV_SOURCE="${PROJECT_DIR}/third_party/kev"
KEV_ENV="${HOME}/.conda/envs/kev-v039"
SHIP="${PROJECT_DIR}/datasets/kev_ship_dataset"
INIT08="${HF_HOME:-${PROJECT_DIR}/third_party/hf-cache}/hub/models--jaredpalmer--kev-0.8b/snapshots/9a45d25eb2ab761841196625383fa1dff0e56c1e"
BASE08="Qwen/Qwen3.5-0.8B-Base"
REV08="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
ROOT="${PROJECT_DIR}/outputs/data_scaling_kev08"
SUBSETS="${ROOT}/subsets"
EVAL_DATA="${SHIP}/test_5k.jsonl"
EVAL_MANIFEST="${SHIP}/test_5k_manifest.json"
EVAL_SCRIPT="${PROJECT_DIR}/scripts/evaluate/run_kev_eval.py"

case "${LANE}" in
  lanea) FRACTIONS=(25) ;;
  laneb) FRACTIONS=(1 2 4 8) ;;
  *) echo "ERROR: unknown lane ${LANE}" >&2; exit 2 ;;
esac

if ! command -v module >/dev/null 2>&1; then source /etc/profile.d/openscow-modules.sh; fi
module purge
module load miniconda3/py313_26.3.2-2
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
source "${KEV_ENV}/bin/activate"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src:${KEV_SOURCE}"
export HF_HOME="${PROJECT_DIR}/third_party/hf-cache"
export HF_HUB_OFFLINE=1

[[ -r "${EVAL_DATA}" && -r "${EVAL_MANIFEST}" && -r "${EVAL_SCRIPT}" ]] || {
  echo "ERROR: missing evaluation input or script" >&2; exit 2
}
[[ -f "${INIT08}/adapter_model.safetensors" && -f "${INIT08}/head.pt" && -f "${INIT08}/training_config.json" ]] || {
  echo "ERROR: incomplete published Kev-0.8B checkpoint ${INIT08}" >&2; exit 2
}
[[ ! -e "${ROOT}/lane_${LANE}_done" ]] || { echo "ERROR: lane ${LANE} already completed" >&2; exit 2; }

python - "${EVAL_DATA}" "${EVAL_MANIFEST}" <<'PY'
import hashlib, json, sys
from pathlib import Path
path, manifest_path = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text())
hash_value = hashlib.sha256(path.read_bytes()).hexdigest()
if manifest.get("subset_records") != 5000 or manifest.get("subset_sha256") != hash_value:
    raise SystemExit("frozen test_5k subset mismatch")
print(f"verified records=5000 sha256={hash_value}")
PY

evaluate() {  # <run_dir> <output_dir>
  [[ ! -e "$2" ]] || { echo "ERROR: refusing to overwrite $2" >&2; exit 2; }
  python -u "${EVAL_SCRIPT}" --run "$1" --data "${EVAL_DATA}" --output "$2" --device cuda
}

printf 'job=%s lane=%s host=%s gpu=%s start=%s\n' \
  "${SLURM_JOB_ID}" "${LANE}" "$(hostname)" "${CUDA_VISIBLE_DEVICES:-unset}" "$(date -Is)"

if [[ "${LANE}" == "laneb" ]]; then
  printf 'job=%s step0 eval start=%s\n' "${SLURM_JOB_ID}" "$(date -Is)"
  evaluate "${INIT08}" "${ROOT}/step0_eval5k"
fi

for FRAC in "${FRACTIONS[@]}"; do
  SUBSET="${SUBSETS}/train_frac${FRAC}.jsonl"
  RUN_DIR="${ROOT}/frac${FRAC}/checkpoint"
  EVAL_DIR="${ROOT}/frac${FRAC}/eval5k"
  [[ -r "${SUBSET}" ]] || { echo "ERROR: missing subset ${SUBSET}" >&2; exit 2; }
  [[ ! -e "${ROOT}/frac${FRAC}" ]] || { echo "ERROR: refusing to overwrite ${ROOT}/frac${FRAC}" >&2; exit 2; }
  python - "${SUBSET}" "${SUBSETS}/subsets_manifest.json" "train_frac${FRAC}" <<'PY'
import hashlib, json, sys
from pathlib import Path
path, manifest_path, name = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
manifest = json.loads(manifest_path.read_text())
entry = manifest["subsets"].get(name)
hash_value = hashlib.sha256(Path(path).read_bytes()).hexdigest()
if entry is None or entry["sha256"] != hash_value:
    raise SystemExit(f"subset mismatch: {name}")
print(f"verified {name} records={entry['records']} sha256={hash_value}")
PY
  printf 'job=%s frac=%s train start=%s\n' "${SLURM_JOB_ID}" "${FRAC}" "$(date -Is)"
  python -u -m kev.train \
    --data "${SUBSET}" --base "${BASE08}" --base_revision "${REV08}" \
    --init_from "${INIT08}" --epochs 2 --lr 2e-5 --lora 16 --batch 2 --accum 4 \
    --dtype bf16 --weights_dtype bf16 --checkpointing 1 --device cuda --out "${RUN_DIR}"
  printf 'job=%s frac=%s eval start=%s\n' "${SLURM_JOB_ID}" "${FRAC}" "$(date -Is)"
  evaluate "${RUN_DIR}" "${EVAL_DIR}"
done

touch "${ROOT}/lane_${LANE}_done"
printf 'job=%s lane=%s finished=%s\n' "${SLURM_JOB_ID}" "${LANE}" "$(date -Is)"
