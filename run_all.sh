#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-configs/default_config.yaml}"
TRACK="${TRACK:-engine_rul}"
DATASET="${DATASET:-CMAPSS}"
CLASSIFIER_MODEL="${CLASSIFIER_MODEL:-LSTMRegressor}"
GEN_MODELS_STRING="${GEN_MODELS:-FlowMatch}"

read -r -a GEN_MODEL_LIST <<< "$GEN_MODELS_STRING"

"$PYTHON_BIN" train_classifier.py \
  --track "$TRACK" \
  --dataset "$DATASET" \
  --model "$CLASSIFIER_MODEL" \
  --config "$CONFIG_PATH"

for GEN_MODEL in "${GEN_MODEL_LIST[@]}"; do
  echo "[run_all] Phase 2 -> ${GEN_MODEL}"
  "$PYTHON_BIN" train_generator.py \
    --track "$TRACK" \
    --dataset "$DATASET" \
    --model "$GEN_MODEL" \
    --config "$CONFIG_PATH"

  RUN_ID="$("$PYTHON_BIN" - <<PY
from src.utils.logger_utils import resolve_run_dir
print(resolve_run_dir("${TRACK}", "${DATASET}", "${GEN_MODEL}").name)
PY
)"

  echo "[run_all] Phase 3 classifier retraining -> ${GEN_MODEL} (${RUN_ID})"
  "$PYTHON_BIN" train_classifier.py \
    --track "$TRACK" \
    --dataset "$DATASET" \
    --model "$CLASSIFIER_MODEL" \
    --gen_model "$GEN_MODEL" \
    --source_run_id "$RUN_ID" \
    --config "$CONFIG_PATH"
done
