#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --job-name=v039-kev4b
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
RUN_ROOT="${PROJECT_DIR}/outputs/v039_kev4b_ship"
DATA_DIR="${RUN_ROOT}/data"
INPUT_SNAPSHOT="${RUN_ROOT}/input_snapshot"
KEV_SOURCE="${PROJECT_DIR}/third_party/kev"
KEV_ENV="${HOME}/.conda/envs/kev-v039"
BASE_MODEL="Qwen/Qwen3.5-4B-Base"
BASE_REVISION="1001bb4d826a52d1f399e183466143f4da7b741b"
KEV_CHECKPOINT="${PROJECT_DIR}/third_party/models/kev-4b"
HF_HOME="${PROJECT_DIR}/third_party/hf-cache"

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/openscow-modules.sh
fi
module purge
module load miniconda3/py313_26.3.2-2
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

[[ -x "${KEV_ENV}/bin/python" ]] || {
  echo "ERROR: missing prepared Kev environment: ${KEV_ENV}" >&2
  exit 2
}
[[ -d "${KEV_SOURCE}/kev" ]] || {
  echo "ERROR: missing Kev source: ${KEV_SOURCE}" >&2
  exit 2
}
[[ -f "${KEV_CHECKPOINT}/head.pt" && -f "${KEV_CHECKPOINT}/adapter_model.safetensors" ]] || {
  echo "ERROR: missing cached Kev checkpoint: ${KEV_CHECKPOINT}" >&2
  exit 2
}
[[ ! -e "${RUN_ROOT}" ]] || {
  echo "ERROR: refusing to overwrite existing run directory: ${RUN_ROOT}" >&2
  exit 2
}
[[ -r "${PROJECT_DIR}/outputs/v036_structured_action_dataset_30000/structured_actions.jsonl" ]] || {
  echo "ERROR: missing structured action dataset" >&2
  exit 2
}
[[ -r "${PROJECT_DIR}/outputs/v038_stage3_language_150k_full/language_samples.jsonl" ]] || {
  echo "ERROR: missing stage-three language checkpoint" >&2
  exit 2
}

source "${KEV_ENV}/bin/activate"
export PYTHONPATH="${PROJECT_DIR}/src:${KEV_SOURCE}"
export HF_HOME
export HF_HUB_OFFLINE=1
KEV_COMMIT="$(cat "${PROJECT_DIR}/scripts/jev/v039-kev-source-revision.txt")"
CODE_COMMIT="$(cat "${PROJECT_DIR}/scripts/slurm/v039-code-revision.txt")"
mkdir -p logs/apps
mkdir -p "${RUN_ROOT}" "${INPUT_SNAPSHOT}"

printf 'job=%s name=%s host=%s start=%s code=%s kev=%s\n' \
  "${SLURM_JOB_ID}" "${SLURM_JOB_NAME}" "$(hostname)" "$(date -Is)" "${CODE_COMMIT}" "${KEV_COMMIT}"
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

# Freeze the currently available stage-three checkpoint before preparing splits.
cp "${PROJECT_DIR}/outputs/v038_stage3_language_150k_full/language_samples.jsonl" \
   "${INPUT_SNAPSHOT}/language_samples.jsonl"
cp "${PROJECT_DIR}/outputs/v038_stage3_language_150k_full/run_manifest.json" \
   "${INPUT_SNAPSHOT}/run_manifest.json"
python "${PROJECT_DIR}/scripts/jev/prepare_kev_ship_data.py" \
  --samples "${INPUT_SNAPSHOT}/language_samples.jsonl" \
  --structured "${PROJECT_DIR}/outputs/v036_structured_action_dataset_30000/structured_actions.jsonl" \
  --output "${DATA_DIR}"

cat > "${RUN_ROOT}/run_config.json" <<EOF
{
  "revision": "v039_kev4b_ship",
  "job_id": "${SLURM_JOB_ID}",
  "code_commit": "${CODE_COMMIT}",
  "kev_commit": "${KEV_COMMIT}",
  "base_model": "${BASE_MODEL}",
  "base_revision": "${BASE_REVISION}",
  "data_manifest": "${DATA_DIR}/manifest.json",
  "device": "cuda",
  "gpu": "${CUDA_VISIBLE_DEVICES:-unset}",
  "training": {
    "init_from": "jaredpalmer/kev-4b@485ace8703592fcf405488b262449990824cfed1",
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
