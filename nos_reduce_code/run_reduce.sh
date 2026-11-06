#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

export ENV_DIR="${ENV_DIR:-/data/nestor_env/envs/nestor}"
export PATH="$ENV_DIR/bin:$PATH"
export HF_HOME="${HF_HOME:-/data/nos_work/cache/huggingface}"

MODEL_PATH="${MODEL_PATH:-/model/nos_disco/advection-diffusion/last.ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/nos_work/results/dict_reduce}"

case "${1:-full}" in
  --check-only)
    python test_time_compute/reduce_dictionary.py \
      --model_path "$MODEL_PATH" --check-only \
      --output_dir "$OUTPUT_DIR/check"
    ;;
  baseline)
    python test_time_compute/reduce_dictionary.py \
      --model_path "$MODEL_PATH" --tasks E_AD_v E_AD_D \
      --strategies full \
      --output_dir "$OUTPUT_DIR/baseline"
    ;;
  smoke)
    python test_time_compute/reduce_dictionary.py \
      --model_path "$MODEL_PATH" --tasks E_AD_v \
      --strategies random frequency cluster --n_operators 64 \
      --random_seeds 0 1 2 --num_samples 30 \
      --output_dir "$OUTPUT_DIR/smoke"
    ;;
  full)
    python test_time_compute/reduce_dictionary.py \
      --model_path "$MODEL_PATH" --tasks E_AD_v E_AD_D \
      --strategies random frequency cluster \
      --n_operators 16 32 64 128 --random_seeds 0 1 2 \
      --beam_width 4 --max_operators 5 --improvement_threshold 5.0 \
      --num_samples 30 \
      --output_dir "$OUTPUT_DIR"
    ;;
  *)
    echo "usage: $0 [--check-only|baseline|smoke|full]" >&2
    exit 1
    ;;
esac
