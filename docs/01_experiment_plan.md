# FlowMatch-PdM Hackathon Pivot Plan

## Scope

The active hackathon scope is now restricted to:

- `track = bearing_rul`
- `datasets = FEMTO, XJTU-SY`
- `downstream regressor = MambaRULRegressor`
- `classical baselines = noise, smote`
- `generator baselines = COTGAN, FaultDiffusion, DiffusionTS, TimeFlow, FlowMatch`
- `FlowMatch ablations = no_tccm, no_prior, no_lap`

Legacy classification and engine-RUL tracks remain in the repository for traceability, but they are out of scope for the final submission.

## Reset State

- `results/` has been archived for a clean restart
- `pipeline_state.json` has been reset to an empty object
- the new execution plan assumes no preserved runs

## Config Guardrails

- Generator batch size is now enforced per dataset through `datasets.<DATASET>.generator_batch_size`
- `FEMTO.generator_batch_size = 32`
- `XJTU-SY.generator_batch_size = 32`
- Classifier batch size remains at the dataset value `64`
- XJTU-SY window size remains `2048`

The XJTU-SY window length was not changed because the completed `FaultDiffusion` run in the ledger already used the current dataset config. Only generator batch size was standardized for the new pivot runs.

## Classical Augmentation

The RUL pipeline now supports two fast classical augmenters:

- `noise`
- `smote`

For RUL tracks, `smote` is implemented as low-RUL minority-window interpolation rather than class-balancing SMOTE. This keeps the CLI unchanged while making the augmentation valid for continuous targets.

## Execution Order

The runner now prioritizes all FlowMatch work first so the core project results land before the traditional augmentations and the heavier generator baselines.

## GPU Split

The current operating plan is:

- `GPU 1`: launch all `FlowMatch` work manually first
- `GPU 0`: use the scripted queue for the remaining non-FlowMatch work

Use separate run tags so the manual FlowMatch lane and the background queue do not collide:

```bash
export CONFIG_PATH=configs/default_config.yaml
export FLOWMATCH_TAG=pivot_rul_gpu1_flowmatch_20260414
export BG_TAG=pivot_rul_gpu0_background_20260414
```

### Step 1

Train the primary `FlowMatch` generators on:

- `bearing_rul / FEMTO`
- `bearing_rul / XJTU-SY`

Manual commands for `GPU 1`:

```bash
#####################################
# Donnnnnnnneeeeee
CUDA_VISIBLE_DEVICES=1 python3 train_generator.py \
  --track bearing_rul \
  --dataset FEMTO \
  --model FlowMatch \
  --run_id "${FLOWMATCH_TAG}_femto_flowmatch" \
  --config "$CONFIG_PATH"
####################################

################################Doneeeeee
CUDA_VISIBLE_DEVICES=1 python3 train_generator.py \
  --track bearing_rul \
  --dataset XJTU-SY \
  --model FlowMatch \
  --run_id "${FLOWMATCH_TAG}_xjtu_sy_flowmatch" \
  --config "$CONFIG_PATH"
################################
```


### Step 2

Train the downstream Mamba regressors for those two primary `FlowMatch` runs.

RUL evaluation writes the following metrics to `evaluation_results/classifier_metrics.json`:

- `rmse`
- `mae`
- `r2`

Manual commands for `GPU 1`:


####################DONEEEEE
```bash
CUDA_VISIBLE_DEVICES=1 python3 train_classifier_aug.py \
  --track bearing_rul \
  --dataset FEMTO \
  --eval_model mamba \
  --gen_model FlowMatch \
  --source_run_id "${FLOWMATCH_TAG}_femto_flowmatch" \
  --output_run_id "${FLOWMATCH_TAG}_femto_flowmatch_mamba_tstr" \
  --config "$CONFIG_PATH"
##########################


###############Done
CUDA_VISIBLE_DEVICES=1 python3 train_classifier_aug.py \
  --track bearing_rul \
  --dataset XJTU-SY \
  --eval_model mamba \
  --gen_model FlowMatch \
  --source_run_id "${FLOWMATCH_TAG}_xjtu_sy_flowmatch" \
  --output_run_id "${FLOWMATCH_TAG}_xjtu_sy_flowmatch_mamba_tstr" \
  --config "$CONFIG_PATH"
#################################
```

### Step 3

Run the FlowMatch ablations on FEMTO:

- `--ablation no_tccm`
- `--ablation no_prior`
- `--ablation no_lap`

Then run the downstream TSTR regressors for each ablation run.

Manual commands for `GPU 1`:

```bash
#######DONEEEEE########
CUDA_VISIBLE_DEVICES=1 python3 train_generator.py \
  --track bearing_rul \
  --dataset FEMTO \
  --model FlowMatch \
  --ablation no_tccm \
  --run_id "${FLOWMATCH_TAG}_femto_flowmatch_no_tccm" \
  --config "$CONFIG_PATH"
#########################

#######DONEEEEE########
CUDA_VISIBLE_DEVICES=1 python3 train_generator.py \
  --track bearing_rul \
  --dataset XJTU-SY \
  --model FlowMatch \
  --ablation no_tccm \
  --run_id "${FLOWMATCH_TAG}_xjtu_sy_flowmatch_no_tccm" \
  --config "$CONFIG_PATH"
#########################


CUDA_VISIBLE_DEVICES=1 python3 train_classifier_aug.py \
  --track bearing_rul \
  --dataset FEMTO \
  --eval_model mamba \
  --gen_model FlowMatch \
  --gen_ablation no_tccm \
  --source_run_id "${FLOWMATCH_TAG}_femto_flowmatch_no_tccm" \
  --output_run_id "${FLOWMATCH_TAG}_femto_flowmatch_no_tccm_mamba_tstr" \
  --config "$CONFIG_PATH"

CUDA_VISIBLE_DEVICES=1 python3 train_generator.py \
  --track bearing_rul \
  --dataset FEMTO \
  --model FlowMatch \
  --ablation no_prior \
  --run_id "${FLOWMATCH_TAG}_femto_flowmatch_no_prior" \
  --config "$CONFIG_PATH"

CUDA_VISIBLE_DEVICES=1 python3 train_classifier_aug.py \
  --track bearing_rul \
  --dataset FEMTO \
  --eval_model mamba \
  --gen_model FlowMatch \
  --gen_ablation no_prior \
  --source_run_id "${FLOWMATCH_TAG}_femto_flowmatch_no_prior" \
  --output_run_id "${FLOWMATCH_TAG}_femto_flowmatch_no_prior_mamba_tstr" \
  --config "$CONFIG_PATH"

CUDA_VISIBLE_DEVICES=1 python3 train_generator.py \
  --track bearing_rul \
  --dataset FEMTO \
  --model FlowMatch \
  --ablation no_lap \
  --run_id "${FLOWMATCH_TAG}_femto_flowmatch_no_lap" \
  --config "$CONFIG_PATH"

CUDA_VISIBLE_DEVICES=1 python3 train_classifier_aug.py \
  --track bearing_rul \
  --dataset FEMTO \
  --eval_model mamba \
  --gen_model FlowMatch \
  --gen_ablation no_lap \
  --source_run_id "${FLOWMATCH_TAG}_femto_flowmatch_no_lap" \
  --output_run_id "${FLOWMATCH_TAG}_femto_flowmatch_no_lap_mamba_tstr" \
  --config "$CONFIG_PATH"
```

### Step 4

Trigger the FEMTO FlowMatch W&B sweep from:

- `configs/sweep_flowmatch_femto.yaml`

Sweep parameters:

- `lr` in `[1e-4, 1e-3]`
- `tccm_lambda` in `[0.01, 1.0]`
- `euler_steps` in `[100, 500]`

Manual commands for `GPU 1`:

```bash
CUDA_VISIBLE_DEVICES=1 wandb sweep configs/sweep_flowmatch_femto.yaml
# then launch the returned agent command:
# CUDA_VISIBLE_DEVICES=1 wandb agent <SWEEP_ID>
```

### Step 5

Run the classical RUL Mamba baselines with:

- noise augmentation on FEMTO
- SMOTE-style minority interpolation on FEMTO
- noise augmentation on XJTU-SY
- SMOTE-style minority interpolation on XJTU-SY

These are the first jobs that belong on the `GPU 0` background queue.

### Step 6

Train the remaining non-FlowMatch generators on both datasets:

- `COTGAN`
- `FaultDiffusion`
- `DiffusionTS`
- `TimeFlow`

### Step 7

Train downstream Mamba regressors using each synthetic generator run from Step 6.

## Entry Points

- Manual `FlowMatch` lane on `GPU 1`: run the command blocks in Steps 1 to 4
- Full scripted queue: `bash final_execution.sh`
- Compatibility runner: `python3 orchestrate.py`
- Tmux launcher: `bash launch.sh`

For the current two-GPU workflow:

```bash
CUDA_VISIBLE_DEVICES=0 RUN_TAG="$BG_TAG" python3 orchestrate.py
```

`orchestrate.py` is now a strict sequential non-FlowMatch queue. Once launched on `GPU 0`, it runs only:

- Step 5 classical Mamba baselines
- Step 6 non-FlowMatch generators
- Step 7 downstream Mamba TSTR for those generators

The queue is strictly one-by-one and does not launch any `FlowMatch`, ablation, or sweep jobs.

The runner uses fixed run IDs derived from the selected run tag and does not rely on pre-populated ledger state.
