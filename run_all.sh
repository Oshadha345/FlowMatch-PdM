#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-configs/default_config.yaml}"
TRACK="${TRACK:-engine_rul}"
DATASET="${DATASET:-CMAPSS}"
CLASSIFIER_MODEL="${CLASSIFIER_MODEL:-LSTMRegressor}"
GEN_MODELS_STRING="${GEN_MODELS:-FlowMatch COTGAN FaultDiffusion}"

read -r -a GEN_MODEL_LIST <<< "$GEN_MODELS_STRING"

echo "════════════════════════════════════════════════════════"
echo " FlowMatch-PdM End-to-End Pipeline"
echo " Track:     $TRACK"
echo " Dataset:   $DATASET"
echo " Generators: ${GEN_MODELS_STRING}"
echo "════════════════════════════════════════════════════════"

# Phase 1: Train baseline classifier
echo "[run_all] Phase 1 -> Baseline classifier ($CLASSIFIER_MODEL)"
"$PYTHON_BIN" train_classifier.py \
  --track "$TRACK" \
  --dataset "$DATASET" \
  --model "$CLASSIFIER_MODEL" \
  --config "$CONFIG_PATH"

declare -A RUN_IDS

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
  RUN_IDS["$GEN_MODEL"]="$RUN_ID"

  echo "[run_all] Phase 3 classifier retraining -> ${GEN_MODEL} (${RUN_ID})"
  "$PYTHON_BIN" train_classifier.py \
    --track "$TRACK" \
    --dataset "$DATASET" \
    --model "$CLASSIFIER_MODEL" \
    --gen_model "$GEN_MODEL" \
    --source_run_id "$RUN_ID" \
    --config "$CONFIG_PATH"
done

echo ""
echo "════════════════════════════════════════════════════════"
echo " Summary"
echo "════════════════════════════════════════════════════════"
printf "%-20s %-30s\n" "Generator" "Run ID"
printf "%-20s %-30s\n" "--------------------" "------------------------------"
for GEN_MODEL in "${GEN_MODEL_LIST[@]}"; do
  printf "%-20s %-30s\n" "$GEN_MODEL" "${RUN_IDS[$GEN_MODEL]}"
done
echo "════════════════════════════════════════════════════════"
echo " Pipeline complete. Check results/$TRACK/$DATASET/ for outputs."
echo "════════════════════════════════════════════════════════"
