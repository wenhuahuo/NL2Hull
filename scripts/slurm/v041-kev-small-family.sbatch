#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --job-name=v041-kev-small-family
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
KEV_SOURCE="${PROJECT_DIR}/third_party/kev"
KEV_ENV="${HOME}/.conda/envs/kev-v039"
HF_HOME="${PROJECT_DIR}/third_party/hf-cache"
DATASET_DIR="${PROJECT_DIR}/datasets/kev_ship_dataset"
ORIGIN_DIR="${PROJECT_DIR}/datasets/kev_origin_dataset"
SUITE_DIR="${KEV_SOURCE}/evals/v7/decision-v7"
ORIGINAL_OUT="${PROJECT_DIR}/outputs/v041_kev2b_original"
KEV08_OUT="${PROJECT_DIR}/outputs/v041_kev0.8b_ship"
KEV2_OUT="${PROJECT_DIR}/outputs/v041_kev2b_ship"
KEV08_INIT="jaredpalmer/kev-0.8b"
BASE08="Qwen/Qwen3.5-0.8B-Base"
BASE08_REVISION="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
BASE2="Qwen/Qwen3.5-2B-Base"
BASE2_REVISION="b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
TRAIN_SCRIPT="${PROJECT_DIR}/scripts/slurm/v041-kev-small-family.sbatch"

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/openscow-modules.sh
fi
module purge
module load miniconda3/py313_26.3.2-2
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

[[ -x "${KEV_ENV}/bin/python" ]] || { echo "ERROR: missing Kev environment" >&2; exit 2; }
[[ -d "${KEV_SOURCE}/kev" ]] || { echo "ERROR: missing Kev source" >&2; exit 2; }
[[ -r "${ORIGIN_DIR}/train.jsonl" && -r "${ORIGIN_DIR}/manifest.json" ]] || {
  echo "ERROR: missing pinned Kev original dataset" >&2; exit 2
}
[[ -r "${SUITE_DIR}/manifest.json" ]] || { echo "ERROR: missing Kev suite manifest" >&2; exit 2; }
for split in train validation test; do
  [[ -r "${DATASET_DIR}/${split}.jsonl" ]] || { echo "ERROR: missing ship dataset ${split}" >&2; exit 2; }
done
for output in "${ORIGINAL_OUT}" "${KEV08_OUT}" "${KEV2_OUT}"; do
  [[ ! -e "${output}" ]] || { echo "ERROR: refusing to overwrite ${output}" >&2; exit 2; }
done

source "${KEV_ENV}/bin/activate"
export PYTHONPATH="${PROJECT_DIR}/src:${KEV_SOURCE}"
export HF_HOME
export HF_ENDPOINT="https://hf-mirror.com"
mkdir -p logs/apps

CODE_COMMIT="unknown"
if git -C "${PROJECT_DIR}" rev-parse HEAD >/dev/null 2>&1; then
  CODE_COMMIT="$(git -C "${PROJECT_DIR}" rev-parse HEAD)"
fi
KEV_COMMIT="$(cat "${PROJECT_DIR}/scripts/jev/v039-kev-source-revision.txt")"
SCRIPT_SHA256="$(sha256sum "${TRAIN_SCRIPT}" | awk '{print $1}')"
ORIGIN_MANIFEST_SHA256="$(sha256sum "${ORIGIN_DIR}/manifest.json" | awk '{print $1}')"
SHIP_MANIFEST_SHA256="$(sha256sum "${DATASET_DIR}/manifest.json" | awk '{print $1}')"

python3 - "${ORIGIN_DIR}/manifest.json" "${ORIGIN_DIR}" <<'PY'
import hashlib, json, sys
from pathlib import Path
manifest_path, root = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text())
for name, info in manifest["files"].items():
    path = root / name
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
        raise SystemExit(f"original Kev dataset hash mismatch: {name}")
print(f"verified original Kev dataset: {manifest['files']['train.jsonl']['records']} train records")
PY
python3 - "${DATASET_DIR}/manifest.json" <<'PY'
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
if manifest.get("revision") != "kev_ship_dataset" or manifest.get("record_count") != 134558:
    raise SystemExit("unexpected ship dataset manifest")
print(f"verified ship dataset: {manifest['record_count']} records")
PY

cat > "${PROJECT_DIR}/outputs/v041-small-family-preflight-${SLURM_JOB_ID}.json" <<EOF
{
  "job_id": "${SLURM_JOB_ID}",
  "code_commit": "${CODE_COMMIT}",
  "script_sha256": "${SCRIPT_SHA256}",
  "kev_source_commit": "${KEV_COMMIT}",
  "origin_dataset": "${ORIGIN_DIR}",
  "origin_manifest_sha256": "${ORIGIN_MANIFEST_SHA256}",
  "ship_dataset": "${DATASET_DIR}",
  "ship_manifest_sha256": "${SHIP_MANIFEST_SHA256}",
  "base_08": "${BASE08}@${BASE08_REVISION}",
  "base_2b": "${BASE2}@${BASE2_REVISION}",
  "schedule": ["kev-2b-original", "kev-0.8b-ship-lora", "kev-2b-ship-lora"]
}
EOF

printf 'job=%s host=%s source=%s code=%s kev=%s\n' \
  "${SLURM_JOB_ID}" "$(hostname)" "${SLURM_JOB_NAME}" "${CODE_COMMIT}" "${KEV_COMMIT}"
python --version
python - <<'PY'
import torch
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
PY

# Stage 1: reproduce the original Kev-style training on Qwen3.5-2B-Base.
python -m kev.train \
  --data "${ORIGIN_DIR}/train.jsonl" \
  --suite "${SUITE_DIR}" \
  --base "${BASE2}" \
  --base_revision "${BASE2_REVISION}" \
  --epochs 2 \
  --lr 5e-5 \
  --lora 16 \
  --batch 2 \
  --accum 4 \
  --dtype bf16 \
  --weights_dtype bf16 \
  --checkpointing 1 \
  --p_none_pair 0.25 \
  --device cuda \
  --out "${ORIGINAL_OUT}"

# Stage 2: fine-tune the released Kev-0.8B checkpoint on the fixed ship dataset.
python -m kev.train \
  --data "${DATASET_DIR}/train.jsonl" \
  --base "${BASE08}" \
  --base_revision "${BASE08_REVISION}" \
  --init_from "${KEV08_INIT}" \
  --epochs 2 \
  --lr 2e-5 \
  --lora 16 \
  --batch 8 \
  --accum 1 \
  --dtype bf16 \
  --weights_dtype bf16 \
  --checkpointing 1 \
  --device cuda \
  --out "${KEV08_OUT}"

# Stage 3: fine-tune the newly trained Kev-2B checkpoint on the fixed ship dataset.
python -m kev.train \
  --data "${DATASET_DIR}/train.jsonl" \
  --base "${BASE2}" \
  --base_revision "${BASE2_REVISION}" \
  --init_from "${ORIGINAL_OUT}" \
  --epochs 2 \
  --lr 2e-5 \
  --lora 16 \
  --batch 2 \
  --accum 4 \
  --dtype bf16 \
  --weights_dtype bf16 \
  --checkpointing 1 \
  --device cuda \
  --out "${KEV2_OUT}"

for model in kev08 kev2; do
  if [[ "${model}" == kev08 ]]; then
    RUN="${KEV08_OUT}"
    PREFIX="kev0.8b"
  else
    RUN="${KEV2_OUT}"
    PREFIX="kev2b"
  fi
  for split in validation test; do
    python -m kev.benchmark \
      --run "${RUN}" \
      --data "${DATASET_DIR}/${split}.jsonl" \
      --device cuda \
      --out "${PROJECT_DIR}/outputs/v041_${PREFIX}_ship_${split}"
  done
done

printf 'job=%s finished=%s outputs=%s,%s,%s\n' \
  "${SLURM_JOB_ID}" "$(date -Is)" "${ORIGINAL_OUT}" "${KEV08_OUT}" "${KEV2_OUT}"
