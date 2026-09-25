#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --job-name=v042-kev-small-resume
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
SHIP="${PROJECT_DIR}/datasets/kev_ship_dataset"
ORIGINAL="${PROJECT_DIR}/outputs/v041_kev2b_original"
INIT08="${HF_HOME}/hub/models--jaredpalmer--kev-0.8b/snapshots/9a45d25eb2ab761841196625383fa1dff0e56c1e"
OUT08="${PROJECT_DIR}/outputs/v042_kev0.8b_ship"
OUT2="${PROJECT_DIR}/outputs/v042_kev2b_ship"
BASE08="Qwen/Qwen3.5-0.8B-Base"
REV08="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
BASE2="Qwen/Qwen3.5-2B-Base"
REV2="b1485b2fa6dfa1287294f269f5fb618e03d52d7c"

if ! command -v module >/dev/null 2>&1; then source /etc/profile.d/openscow-modules.sh; fi
module purge
module load miniconda3/py313_26.3.2-2
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
[[ -x "${KEV_ENV}/bin/python" && -d "${KEV_SOURCE}/kev" ]] || { echo 'missing Kev environment or source' >&2; exit 2; }
for f in "${SHIP}/train.jsonl" "${SHIP}/validation.jsonl" "${SHIP}/test.jsonl" "${SHIP}/manifest.json" \
         "${INIT08}/adapter_model.safetensors" "${INIT08}/head.pt" "${INIT08}/training_config.json" \
         "${ORIGINAL}/adapter_model.safetensors" "${ORIGINAL}/head.pt" "${ORIGINAL}/training_config.json" \
         "${ORIGINAL}/training_metrics.json"; do
  [[ -s "${f}" ]] || { echo "missing ${f}" >&2; exit 2; }
done
for d in "${OUT08}" "${OUT2}" \
         "${PROJECT_DIR}/outputs/v042_kev0.8b_ship_validation" "${PROJECT_DIR}/outputs/v042_kev0.8b_ship_test" \
         "${PROJECT_DIR}/outputs/v042_kev2b_ship_validation" "${PROJECT_DIR}/outputs/v042_kev2b_ship_test"; do
  [[ ! -e "${d}" ]] || { echo "refusing to overwrite ${d}" >&2; exit 2; }
done
python3 - "${SHIP}/manifest.json" "${ORIGINAL}/training_config.json" "${INIT08}/training_config.json" <<'PY'
import json, sys
from pathlib import Path
ship, original, init08 = (json.loads(Path(p).read_text()) for p in sys.argv[1:])
assert ship['revision'] == 'kev_ship_dataset' and ship['record_count'] == 134558
assert original['base_revision'] == 'b1485b2fa6dfa1287294f269f5fb618e03d52d7c'
assert original['args']['epochs'] == 2 and original['args']['base'] == 'Qwen/Qwen3.5-2B-Base'
assert init08['base_revision'] == 'dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68'
assert init08['args']['base'] == 'Qwen/Qwen3.5-0.8B-Base'
print('verified ship dataset and completed 2B / published 0.8B checkpoint metadata')
PY

source "${KEV_ENV}/bin/activate"
export PYTHONPATH="${PROJECT_DIR}/src:${KEV_SOURCE}"
export HF_HOME HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
printf 'job=%s host=%s time=%s gpu=%s original=%s\n' \
  "${SLURM_JOB_ID}" "$(hostname)" "$(date -Is)" "${CUDA_VISIBLE_DEVICES:-unset}" "${ORIGINAL}"
python - <<'PY'
import torch
print(f'torch={torch.__version__} cuda={torch.cuda.is_available()}')
if not torch.cuda.is_available(): raise SystemExit('CUDA unavailable')
PY

# Resume after job 4505: original Kev-2B is complete; do not retrain or overwrite it.
python -m kev.train \
  --data "${SHIP}/train.jsonl" --base "${BASE08}" --base_revision "${REV08}" \
  --init_from "${INIT08}" --epochs 2 --lr 2e-5 --lora 16 --batch 2 --accum 4 \
  --dtype bf16 --weights_dtype bf16 --checkpointing 1 --device cuda --out "${OUT08}"
python -m kev.train \
  --data "${SHIP}/train.jsonl" --base "${BASE2}" --base_revision "${REV2}" \
  --init_from "${ORIGINAL}" --epochs 2 --lr 2e-5 --lora 16 --batch 2 --accum 4 \
  --dtype bf16 --weights_dtype bf16 --checkpointing 1 --device cuda --out "${OUT2}"

for model in 'kev0.8b' 'kev2b'; do
  if [[ "${model}" == 'kev0.8b' ]]; then run="${OUT08}"; else run="${OUT2}"; fi
  for split in validation test; do
    python -m kev.benchmark --run "${run}" --data "${SHIP}/${split}.jsonl" \
      --device cuda --out "${PROJECT_DIR}/outputs/v042_${model}_ship_${split}"
  done
done
printf 'job=%s finished=%s outputs=%s,%s\n' "${SLURM_JOB_ID}" "$(date -Is)" "${OUT08}" "${OUT2}"
