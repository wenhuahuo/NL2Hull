#!/bin/bash
#SBATCH --account=internal
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --job-name=v040-qwen3-embedding
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
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
PROJECT=/data/home/wenhua_huo/Documents/NurbsShipGen
ENV_PY="$HOME/.conda/envs/kev-v039/bin/python"
SCRIPT="$PROJECT/visulization/v040_qwen3_embedding_distribution/prepare_and_embed.py"
INPUT="$PROJECT/outputs/v038_stage3_language_150k_cleaned_rollback/language_samples.jsonl"
OUTPUT="$PROJECT/outputs/v040_qwen3_embedding_distribution"
test -x "$ENV_PY"
test -r "$INPUT"
test -r "$SCRIPT"
test ! -e "$OUTPUT"
printf 'job=%s host=%s start=%s input=%s output=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -Is)" "$INPUT" "$OUTPUT"
"$ENV_PY" "$SCRIPT" --input "$INPUT" --output "$OUTPUT" --batch-size 32
printf 'job=%s finished=%s\n' "$SLURM_JOB_ID" "$(date -Is)"
