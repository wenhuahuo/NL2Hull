#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --job-name=v041-kev4b-cleaned
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/data/home/wenhua_huo/Documents/NurbsShipGen
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR="/data/home/wenhua_huo/Documents/NurbsShipGen"
REVISION="v041_kev4b_ship_cleaned"
RUN_ROOT="${PROJECT_DIR}/outputs/${REVISION}"
DATA_DIR="${RUN_ROOT}/data"
INPUT_SNAPSHOT="${RUN_ROOT}/input_snapshot"
SOURCE_SNAPSHOT="${PROJECT_DIR}/outputs/${REVISION}_input_snapshot"
KEV_SOURCE="${PROJECT_DIR}/third_party/kev"
KEV_ENV="${HOME}/.conda/envs/kev-v039"
BASE_MODEL="Qwen/Qwen3.5-4B-Base"
BASE_REVISION="1001bb4d826a52d1f399e183466143f4da7b741b"
KEV_CHECKPOINT="${PROJECT_DIR}/third_party/models/kev-4b"
HF_HOME="${PROJECT_DIR}/third_party/hf-cache"
STRUCTURED="${PROJECT_DIR}/outputs/v036_structured_action_dataset_30000/structured_actions.jsonl"
PREPARE_SCRIPT="${PROJECT_DIR}/scripts/jev/prepare_kev_ship_data.py"
TRAIN_SCRIPT="${PROJECT_DIR}/scripts/slurm/v041-kev4b-ship-cleaned.sbatch"

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/openscow-modules.sh
fi
module purge
module load miniconda3/py313_26.3.2-2
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

[[ -x "${KEV_ENV}/bin/python" ]] || { echo "ERROR: missing Kev environment" >&2; exit 2; }
[[ -d "${KEV_SOURCE}/kev" ]] || { echo "ERROR: missing Kev source" >&2; exit 2; }
[[ -f "${KEV_CHECKPOINT}/head.pt" && -f "${KEV_CHECKPOINT}/adapter_model.safetensors" ]] || {
  echo "ERROR: missing cached Kev checkpoint" >&2; exit 2
}
[[ ! -e "${RUN_ROOT}" ]] || { echo "ERROR: refusing to overwrite ${RUN_ROOT}" >&2; exit 2; }
[[ -r "${SOURCE_SNAPSHOT}/language_samples.jsonl" ]] || { echo "ERROR: missing cleaned language snapshot" >&2; exit 2; }
[[ -r "${SOURCE_SNAPSHOT}/selection_manifest.json" ]] || { echo "ERROR: missing selection manifest" >&2; exit 2; }
[[ -r "${STRUCTURED}" && -r "${PREPARE_SCRIPT}" ]] || { echo "ERROR: missing Kev input files" >&2; exit 2; }

source "${KEV_ENV}/bin/activate"
export PYTHONPATH="${PROJECT_DIR}/src:${KEV_SOURCE}"
export HF_HOME
export HF_HUB_OFFLINE=1
mkdir -p logs/apps
mkdir -p "${RUN_ROOT}" "${INPUT_SNAPSHOT}"

CODE_COMMIT="unknown"
if git -C "${PROJECT_DIR}" rev-parse HEAD >/dev/null 2>&1; then
  CODE_COMMIT="$(git -C "${PROJECT_DIR}" rev-parse HEAD)"
fi
KEV_COMMIT="$(cat "${PROJECT_DIR}/scripts/jev/v039-kev-source-revision.txt")"
SCRIPT_SHA256="$(sha256sum "${TRAIN_SCRIPT}" | awk '{print $1}')"
PREPARE_SHA256="$(sha256sum "${PREPARE_SCRIPT}" | awk '{print $1}')"
STRUCTURED_SHA256="$(sha256sum "${STRUCTURED}" | awk '{print $1}')"

printf 'job=%s name=%s host=%s start=%s revision=%s code=%s kev=%s\n' \
  "${SLURM_JOB_ID}" "${SLURM_JOB_NAME}" "$(hostname)" "$(date -Is)" \
  "${REVISION}" "${CODE_COMMIT}" "${KEV_COMMIT}"
printf 'account=%s partition=%s qos=%s cpus=%s gpu=%s\n' \
  "${SLURM_JOB_ACCOUNT}" "${SLURM_JOB_PARTITION}" "${SLURM_JOB_QOS}" \
  "${SLURM_CPUS_PER_TASK}" "${CUDA_VISIBLE_DEVICES:-unset}"
python --version
python - <<'PY'
import torch
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
PY

cp "${SOURCE_SNAPSHOT}/language_samples.jsonl" "${INPUT_SNAPSHOT}/language_samples.jsonl"
cp "${SOURCE_SNAPSHOT}/selection_manifest.json" "${INPUT_SNAPSHOT}/selection_manifest.json"
if [[ -f "${SOURCE_SNAPSHOT}/cleaning_manifest.json" ]]; then
  cp "${SOURCE_SNAPSHOT}/cleaning_manifest.json" "${INPUT_SNAPSHOT}/cleaning_manifest.json"
fi
python - "${INPUT_SNAPSHOT}/language_samples.jsonl" "${INPUT_SNAPSHOT}/selection_manifest.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
samples, manifest = map(Path, sys.argv[1:])
actual = hashlib.sha256(samples.read_bytes()).hexdigest()
expected = json.loads(manifest.read_text())["language_samples_sha256"]
if actual != expected:
    raise SystemExit(f"snapshot hash mismatch: {actual} != {expected}")
print(f"verified language snapshot sha256={actual}")
PY

python "${PREPARE_SCRIPT}" \
  --samples "${INPUT_SNAPSHOT}/language_samples.jsonl" \
  --structured "${STRUCTURED}" \
  --output "${DATA_DIR}"

cat > "${RUN_ROOT}/run_config.json" <<EOF
{
  "revision": "${REVISION}",
  "job_id": "${SLURM_JOB_ID}",
  "code_commit": "${CODE_COMMIT}",
  "script_sha256": "${SCRIPT_SHA256}",
  "prepare_script_sha256": "${PREPARE_SHA256}",
  "kev_commit": "${KEV_COMMIT}",
  "base_model": "${BASE_MODEL}",
  "base_revision": "${BASE_REVISION}",
  "input_snapshot": "${INPUT_SNAPSHOT}/language_samples.jsonl",
  "data_manifest": "${DATA_DIR}/manifest.json",
  "structured_sha256": "${STRUCTURED_SHA256}",
  "device": "cuda",
  "gpu": "${CUDA_VISIBLE_DEVICES:-unset}",
  "training": {
    "init_from": "${KEV_CHECKPOINT}",
    "epochs": 2,
    "lr": 0.00002,
    "batch": 1,
    "accum": 8,
    "dtype": "bf16",
    "weights_dtype": "bf16",
    "checkpointing": 1,
    "lora": 16
  }
}
EOF

for split in validation test; do
  python -m kev.benchmark \
    --run "${KEV_CHECKPOINT}" \
    --data "${DATA_DIR}/${split}.jsonl" \
    --device cuda \
    --out "${RUN_ROOT}/baseline_${split}"
done

python -m kev.train \
  --data "${DATA_DIR}/train.jsonl" \
  --base "${BASE_MODEL}" \
  --base_revision "${BASE_REVISION}" \
  --init_from "${KEV_CHECKPOINT}" \
  --epochs 2 \
  --lr 2e-5 \
  --batch 1 \
  --accum 8 \
  --dtype bf16 \
  --weights_dtype bf16 \
  --checkpointing 1 \
  --device cuda \
  --out "${RUN_ROOT}/checkpoint"

for split in validation test; do
  python -m kev.benchmark \
    --run "${RUN_ROOT}/checkpoint" \
    --data "${DATA_DIR}/${split}.jsonl" \
    --device cuda \
    --out "${RUN_ROOT}/trained_${split}"
done

python "${PROJECT_DIR}/scripts/jev/compare_kev_reports.py" \
  --baseline "${RUN_ROOT}/baseline_test/report.json" \
  --trained "${RUN_ROOT}/trained_test/report.json" \
  --output "${RUN_ROOT}/test_comparison.json"
python "${PROJECT_DIR}/scripts/jev/compare_kev_reports.py" \
  --baseline "${RUN_ROOT}/baseline_validation/report.json" \
  --trained "${RUN_ROOT}/trained_validation/report.json" \
  --output "${RUN_ROOT}/validation_comparison.json"

printf 'job=%s finished=%s run_root=%s\n' "${SLURM_JOB_ID}" "$(date -Is)" "${RUN_ROOT}"
