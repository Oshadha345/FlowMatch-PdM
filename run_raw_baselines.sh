#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_PATH="${CONFIG_PATH:-configs/default_config.yaml}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
RUN_TAG="${RUN_TAG:-raw_baselines_20260415}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter not found. Activate the flowmatch_pdm environment first." >&2
  exit 1
fi

run_cmd() {
  printf '\n[exec]'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

echo "============================================================"
echo "FlowMatch-PdM Raw Baseline Evaluators"
echo "Track: bearing_rul"
echo "Datasets: FEMTO, XJTU-SY"
echo "GPU: ${GPU_ID}"
echo "Run tag: ${RUN_TAG}"
echo "Config: ${CONFIG_PATH}"
echo "============================================================"

echo
echo "### FEMTO Raw Baselines"
run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_classifier_aug.py \
  --track bearing_rul \
  --dataset FEMTO \
  --model raw \
  --eval_model lstm \
  --epochs 20 \
  --output_run_id "${RUN_TAG}_femto_lstm_raw" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_classifier_aug.py \
  --track bearing_rul \
  --dataset FEMTO \
  --model raw \
  --eval_model cnn1d \
  --epochs 20 \
  --output_run_id "${RUN_TAG}_femto_cnn1d_raw" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_classifier_aug.py \
  --track bearing_rul \
  --dataset FEMTO \
  --model raw \
  --eval_model transformer \
  --epochs 20 \
  --output_run_id "${RUN_TAG}_femto_transformer_raw" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_classifier_aug.py \
  --track bearing_rul \
  --dataset FEMTO \
  --model raw \
  --eval_model mamba \
  --epochs 20 \
  --output_run_id "${RUN_TAG}_femto_mamba_raw" \
  --config "$CONFIG_PATH"

echo
echo "### XJTU-SY Raw Baselines"
run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_classifier_aug.py \
  --track bearing_rul \
  --dataset XJTU-SY \
  --model raw \
  --eval_model lstm \
  --epochs 20 \
  --output_run_id "${RUN_TAG}_xjtu_sy_lstm_raw" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_classifier_aug.py \
  --track bearing_rul \
  --dataset XJTU-SY \
  --model raw \
  --eval_model cnn1d \
  --epochs 20 \
  --output_run_id "${RUN_TAG}_xjtu_sy_cnn1d_raw" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_classifier_aug.py \
  --track bearing_rul \
  --dataset XJTU-SY \
  --model raw \
  --eval_model transformer \
  --epochs 20 \
  --output_run_id "${RUN_TAG}_xjtu_sy_transformer_raw" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_classifier_aug.py \
  --track bearing_rul \
  --dataset XJTU-SY \
  --model raw \
  --eval_model mamba \
  --epochs 20 \
  --output_run_id "${RUN_TAG}_xjtu_sy_mamba_raw" \
  --config "$CONFIG_PATH"
