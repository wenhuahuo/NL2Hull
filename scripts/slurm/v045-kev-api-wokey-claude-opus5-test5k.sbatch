#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --chdir=/data/home/wenhua_huo/Documents/NurbsShipGen
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1
PROJECT_DIR=/data/home/wenhua_huo/Documents/NurbsShipGen
KEV_SOURCE="${PROJECT_DIR}/third_party/kev"
KEV_ENV="${HOME}/.conda/envs/kev-v039"
DATA_FILE="${PROJECT_DIR}/datasets/kev_ship_dataset/test_5k.jsonl"
DATA_MANIFEST="${PROJECT_DIR}/datasets/kev_ship_dataset/test_5k_manifest.json"
EVAL_SCRIPT="${PROJECT_DIR}/scripts/evaluate/run_llm_eval.py"
OUTPUT_DIR="${PROJECT_DIR}/outputs/v045_kev_api_wokey_claude_opus5_test_5k"
MODEL="claude-opus-5"
PROVIDER="wokey"
THINKING="low"

if ! command -v module >/dev/null 2>&1; then source /etc/profile.d/openscow-modules.sh; fi
module purge
module load miniconda3/py313_26.3.2-2
module load pi/0.87.0
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
cd "${PROJECT_DIR}"

[[ -r "${DATA_FILE}" && -r "${DATA_MANIFEST}" && -r "${EVAL_SCRIPT}" && -r .env ]] || {
  echo "ERROR: missing evaluation input, script, or .env" >&2; exit 2
}
[[ -d "${KEV_SOURCE}/kev" ]] || { echo "ERROR: missing Kev source" >&2; exit 2; }
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "ERROR: refusing to overwrite ${OUTPUT_DIR}" >&2; exit 2; }
command -v pi >/dev/null || { echo "ERROR: pi is unavailable" >&2; exit 2; }

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

set -a
source .env
set +a
RUNTIME_DIR="${PROJECT_DIR}/logs/apps/pi-runtime-${PROVIDER}-${MODEL//[^A-Za-z0-9]/_}-${SLURM_JOB_ID}"
mkdir -p "${RUNTIME_DIR}" "${PROJECT_DIR}/logs/apps"
cleanup() { rm -rf -- "${RUNTIME_DIR}"; }
trap cleanup EXIT
cat > "${RUNTIME_DIR}/models.json" <<'JSON'
{
  "providers": {
    "wokey": {
      "baseUrl": "https://api.wokey.ai/v1",
      "api": "openai-completions",
      "apiKey": "$WOKEY_API_KEY",
      "models": [
        {
          "id": "claude-opus-5",
          "name": "Claude Opus 5",
          "api": "anthropic-messages",
          "baseUrl": "https://api.wokey.ai",
          "reasoning": true,
          "input": ["text", "image"],
          "contextWindow": 1000000,
          "maxTokens": 128000,
          "thinkingLevelMap": {
            "off": null,
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh",
            "max": "max"
          },
          "compat": {
            "forceAdaptiveThinking": true,
            "supportsTemperature": false,
            "supportsStrictTools": true,
            "supportsCacheControlOnTools": false,
            "supportsLongCacheRetention": false
          }
        }
      ]
    }
  }
}
JSON
export PI_CODING_AGENT_DIR="${RUNTIME_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}/scripts/evaluate:${KEV_SOURCE}"
printf 'job=%s host=%s model=%s provider=%s thinking=%s records=%s data_sha256=%s start=%s\n' \
  "${SLURM_JOB_ID}" "$(hostname)" "${MODEL}" "${PROVIDER}" "${THINKING}" \
  "$(wc -l < "${DATA_FILE}")" "$(sha256sum "${DATA_FILE}" | cut -d' ' -f1)" "$(date -Is)"
"${KEV_ENV}/bin/python" -u "${EVAL_SCRIPT}" \
  --data "${DATA_FILE}" \
  --output "${OUTPUT_DIR}" \
  --model "${MODEL}" \
  --provider "${PROVIDER}" \
  --thinking "${THINKING}" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --timeout 180 \
  --retries 2
printf 'job=%s finished=%s output=%s\n' "${SLURM_JOB_ID}" "$(date -Is)" "${OUTPUT_DIR}"
