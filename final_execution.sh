#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${CONFIG_PATH:-configs/default_config.yaml}"
RUN_TAG="${RUN_TAG:-pivot_rul_mamba_20260414}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

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
echo "FlowMatch-PdM Hard Pivot Execution"
echo "Track: bearing_rul"
echo "Datasets: FEMTO, XJTU-SY"
echo "Downstream regressor: MambaRULRegressor"
echo "Config: ${CONFIG_PATH}"
echo "GPU: ${GPU_ID}"
echo "Run tag: ${RUN_TAG}"
echo "============================================================"

echo
echo "### Priority 1: FlowMatch Generation & Evaluation"
run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_generator.py \
  --track bearing_rul \
  --dataset FEMTO \
  --model FlowMatch \
  --run_id "${RUN_TAG}_femto_flowmatch" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_generator.py \
  --track bearing_rul \
  --dataset XJTU-SY \
  --model FlowMatch \
  --run_id "${RUN_TAG}_xjtu_sy_flowmatch" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_classifier_aug.py \
  --track bearing_rul \
  --dataset FEMTO \
  --eval_model mamba \
  --gen_model FlowMatch \
  --source_run_id "${RUN_TAG}_femto_flowmatch" \
  --output_run_id "${RUN_TAG}_femto_flowmatch_mamba_tstr" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_classifier_aug.py \
  --track bearing_rul \
  --dataset XJTU-SY \
  --eval_model mamba \
  --gen_model FlowMatch \
  --source_run_id "${RUN_TAG}_xjtu_sy_flowmatch" \
  --output_run_id "${RUN_TAG}_xjtu_sy_flowmatch_mamba_tstr" \
  --config "$CONFIG_PATH"

echo
echo "### Priority 2: Traditional Baselines"
run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_classifier.py \
  --track bearing_rul \
  --dataset FEMTO \
  --eval_model mamba \
  --aug noise \
  --run_id "${RUN_TAG}_femto_mamba_noise" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_classifier.py \
  --track bearing_rul \
  --dataset FEMTO \
  --eval_model mamba \
  --aug smote \
  --run_id "${RUN_TAG}_femto_mamba_smote" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_classifier.py \
  --track bearing_rul \
  --dataset XJTU-SY \
  --eval_model mamba \
  --aug noise \
  --run_id "${RUN_TAG}_xjtu_sy_mamba_noise" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_classifier.py \
  --track bearing_rul \
  --dataset XJTU-SY \
  --eval_model mamba \
  --aug smote \
  --run_id "${RUN_TAG}_xjtu_sy_mamba_smote" \
  --config "$CONFIG_PATH"

echo
echo "### Priority 3: Heavy Generative Baselines"
run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_generator.py \
  --track bearing_rul \
  --dataset FEMTO \
  --model COTGAN \
  --run_id "${RUN_TAG}_femto_cotgan" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_generator.py \
  --track bearing_rul \
  --dataset FEMTO \
  --model FaultDiffusion \
  --run_id "${RUN_TAG}_femto_faultdiffusion" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_generator.py \
  --track bearing_rul \
  --dataset FEMTO \
  --model TimeFlow \
  --run_id "${RUN_TAG}_femto_timeflow" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_generator.py \
  --track bearing_rul \
  --dataset FEMTO \
  --model DiffusionTS \
  --run_id "${RUN_TAG}_femto_diffusionts" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_generator.py \
  --track bearing_rul \
  --dataset XJTU-SY \
  --model COTGAN \
  --run_id "${RUN_TAG}_xjtu_sy_cotgan" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_generator.py \
  --track bearing_rul \
  --dataset XJTU-SY \
  --model FaultDiffusion \
  --run_id "${RUN_TAG}_xjtu_sy_faultdiffusion" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_generator.py \
  --track bearing_rul \
  --dataset XJTU-SY \
  --model TimeFlow \
  --run_id "${RUN_TAG}_xjtu_sy_timeflow" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_generator.py \
  --track bearing_rul \
  --dataset XJTU-SY \
  --model DiffusionTS \
  --run_id "${RUN_TAG}_xjtu_sy_diffusionts" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_classifier_aug.py \
  --track bearing_rul \
  --dataset FEMTO \
  --eval_model mamba \
  --gen_model COTGAN \
  --source_run_id "${RUN_TAG}_femto_cotgan" \
  --output_run_id "${RUN_TAG}_femto_cotgan_mamba_tstr" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_classifier_aug.py \
  --track bearing_rul \
  --dataset FEMTO \
  --eval_model mamba \
  --gen_model FaultDiffusion \
  --source_run_id "${RUN_TAG}_femto_faultdiffusion" \
  --output_run_id "${RUN_TAG}_femto_faultdiffusion_mamba_tstr" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_classifier_aug.py \
  --track bearing_rul \
  --dataset FEMTO \
  --eval_model mamba \
  --gen_model TimeFlow \
  --source_run_id "${RUN_TAG}_femto_timeflow" \
  --output_run_id "${RUN_TAG}_femto_timeflow_mamba_tstr" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_classifier_aug.py \
  --track bearing_rul \
  --dataset FEMTO \
  --eval_model mamba \
  --gen_model DiffusionTS \
  --source_run_id "${RUN_TAG}_femto_diffusionts" \
  --output_run_id "${RUN_TAG}_femto_diffusionts_mamba_tstr" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_classifier_aug.py \
  --track bearing_rul \
  --dataset XJTU-SY \
  --eval_model mamba \
  --gen_model COTGAN \
  --source_run_id "${RUN_TAG}_xjtu_sy_cotgan" \
  --output_run_id "${RUN_TAG}_xjtu_sy_cotgan_mamba_tstr" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_classifier_aug.py \
  --track bearing_rul \
  --dataset XJTU-SY \
  --eval_model mamba \
  --gen_model FaultDiffusion \
  --source_run_id "${RUN_TAG}_xjtu_sy_faultdiffusion" \
  --output_run_id "${RUN_TAG}_xjtu_sy_faultdiffusion_mamba_tstr" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_classifier_aug.py \
  --track bearing_rul \
  --dataset XJTU-SY \
  --eval_model mamba \
  --gen_model TimeFlow \
  --source_run_id "${RUN_TAG}_xjtu_sy_timeflow" \
  --output_run_id "${RUN_TAG}_xjtu_sy_timeflow_mamba_tstr" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_classifier_aug.py \
  --track bearing_rul \
  --dataset XJTU-SY \
  --eval_model mamba \
  --gen_model DiffusionTS \
  --source_run_id "${RUN_TAG}_xjtu_sy_diffusionts" \
  --output_run_id "${RUN_TAG}_xjtu_sy_diffusionts_mamba_tstr" \
  --config "$CONFIG_PATH"

echo
echo "### Priority 4: The Proof"
run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_generator.py \
  --track bearing_rul \
  --dataset FEMTO \
  --model FlowMatch \
  --ablation no_tccm \
  --run_id "${RUN_TAG}_femto_flowmatch_no_tccm" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_classifier_aug.py \
  --track bearing_rul \
  --dataset FEMTO \
  --eval_model mamba \
  --gen_model FlowMatch \
  --gen_ablation no_tccm \
  --source_run_id "${RUN_TAG}_femto_flowmatch_no_tccm" \
  --output_run_id "${RUN_TAG}_femto_flowmatch_no_tccm_mamba_tstr" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_generator.py \
  --track bearing_rul \
  --dataset FEMTO \
  --model FlowMatch \
  --ablation no_prior \
  --run_id "${RUN_TAG}_femto_flowmatch_no_prior" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_classifier_aug.py \
  --track bearing_rul \
  --dataset FEMTO \
  --eval_model mamba \
  --gen_model FlowMatch \
  --gen_ablation no_prior \
  --source_run_id "${RUN_TAG}_femto_flowmatch_no_prior" \
  --output_run_id "${RUN_TAG}_femto_flowmatch_no_prior_mamba_tstr" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_generator.py \
  --track bearing_rul \
  --dataset FEMTO \
  --model FlowMatch \
  --ablation no_lap \
  --run_id "${RUN_TAG}_femto_flowmatch_no_lap" \
  --config "$CONFIG_PATH"

run_cmd env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" train_classifier_aug.py \
  --track bearing_rul \
  --dataset FEMTO \
  --eval_model mamba \
  --gen_model FlowMatch \
  --gen_ablation no_lap \
  --source_run_id "${RUN_TAG}_femto_flowmatch_no_lap" \
  --output_run_id "${RUN_TAG}_femto_flowmatch_no_lap_mamba_tstr" \
  --config "$CONFIG_PATH"

if [[ "${RUN_WANDB_SWEEP:-0}" == "1" ]]; then
  run_cmd wandb sweep configs/sweep_flowmatch_femto.yaml
  echo "[next] Launch the returned sweep with: wandb agent <SWEEP_ID>"
else
  echo "[skip] W&B sweep disabled. Set RUN_WANDB_SWEEP=1 to enable it."
fi
