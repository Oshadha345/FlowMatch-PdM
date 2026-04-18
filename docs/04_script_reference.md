# Script Reference

This reference reflects the active FEMTO/XJTU-SY RUL pivot.

## `final_execution.sh`

Purpose:

- execute the full hackathon pivot plan
- use deterministic run IDs for all new pivot jobs
- run FlowMatch and Mamba downstream training before the heavier baselines

Usage:

```bash
bash final_execution.sh
```

Environment overrides:

- `PYTHON_BIN`
- `CONFIG_PATH`
- `CUDA_VISIBLE_DEVICES`
- `RUN_TAG`

## `non_flowmatch_execution.sh`

Purpose:

- execute only the non-FlowMatch background queue
- run the classical Mamba baselines first
- run non-FlowMatch generators next
- run downstream Mamba TSTR for those generators after that

Usage:

```bash
bash non_flowmatch_execution.sh
```

Environment overrides:

- `PYTHON_BIN`
- `CONFIG_PATH`
- `CUDA_VISIBLE_DEVICES`
- `RUN_TAG`

## `train_classifier.py`

Purpose:

- train a baseline RUL regressor
- train classically augmented RUL regressors
- train generator-augmented downstream RUL regressors
- auto-evaluate and write regression metrics

CLI:

```bash
python train_classifier.py \
  --track bearing_rul \
  --dataset <FEMTO|XJTU-SY> \
  --eval_model <baseline|mamba|MambaRegressor> \
  [--run_id <OUTPUT_RUN_ID>] \
  [--aug <none|noise|smote>] \
  [--gen_model <COTGAN|FaultDiffusion|DiffusionTS|TimeFlow|FlowMatch>] \
  [--source_run_id <GENERATOR_RUN_ID>] \
  [--gen_ablation <none|no_prior|no_tccm|no_lap>] \
  [--config configs/default_config.yaml]
```

Notes:

- `baseline` resolves to `MambaRegressor` for the active RUL pivot
- `--eval_model mamba` is the preferred explicit form for new runs
- classifier training still uses `epochs = 500` as the hard cap, but stops early on `val_rmse` plateau or when the dataset RMSE target is reached
- `--aug smote` on `bearing_rul` uses minority-window interpolation instead of class-balancing SMOTE
- RUL evaluation writes `rmse`, `mae`, and `r2` to `evaluation_results/classifier_metrics.json`

## `train_classifier_aug.py`

Purpose:

- compatibility wrapper around `train_classifier.py`
- preferred in the pivot plan for generator-augmented downstream retraining

Supported remaps:

- legacy `--run_id` -> `--source_run_id`
- `--output_run_id` -> `--run_id`

Example:

```bash
python train_classifier_aug.py \
  --track bearing_rul \
  --dataset FEMTO \
  --model baseline \
  --gen_model FlowMatch \
  --source_run_id pivot_rul_20260414_femto_flowmatch \
  --output_run_id pivot_rul_20260414_femto_flowmatch_tstr
```

## `train_generator.py`

Purpose:

- train a generator on the minority subset
- auto-evaluate the generator after training
- support FlowMatch ablations and W&B sweeps

CLI:

```bash
python train_generator.py \
  --track bearing_rul \
  --dataset <FEMTO|XJTU-SY> \
  --model <DiffusionTS|TimeFlow|COTGAN|FaultDiffusion|FlowMatch> \
  [--run_id <OUTPUT_RUN_ID>] \
  [--ablation <none|no_prior|no_tccm|no_lap>] \
  [--use_wandb] \
  [--lr <FLOAT>] \
  [--batch_size <INT>] \
  [--epochs <INT>] \
  [--euler_steps <INT>] \
  [--mamba_d_model <INT>] \
  [--mamba_d_state <INT>] \
  [--tccm_lambda <FLOAT>] \
  [--lap_threshold <FLOAT>] \
  [--config configs/default_config.yaml]
```

Notes:

- only `bearing_rul / FEMTO|XJTU-SY` is supported in the active pivot
- only `COTGAN`, `FaultDiffusion`, `DiffusionTS`, `TimeFlow`, and `FlowMatch` are supported
- `datasets.FEMTO.generator_batch_size` and `datasets.XJTU-SY.generator_batch_size` enforce a generator batch size of `32`

## `run_all.sh`

Purpose:

- compatibility entrypoint

Behavior:

- delegates directly to `final_execution.sh`

## `orchestrate.py`

Purpose:

- execute the strict sequential non-FlowMatch queue
- serve as the `GPU 0` background runner while `FlowMatch` is launched manually on another GPU

Behavior:

- delegates directly to `non_flowmatch_execution.sh`
- runs jobs one by one in the order defined there
- never launches `FlowMatch`, its ablations, or the sweep

## `launch.sh`

Purpose:

- launch the pivot run inside `tmux`

Behavior:

- activates `flowmatch_pdm`
- runs `bash final_execution.sh`
- writes output to `logs/final_execution_<timestamp>.log`

## `configs/sweep_flowmatch_femto.yaml`

Purpose:

- W&B Bayesian sweep for `FlowMatch` on `bearing_rul / FEMTO`

Usage:

```bash
wandb sweep configs/sweep_flowmatch_femto.yaml
wandb agent <SWEEP_ID>
```

Sweep parameters:

- `lr`
- `tccm_lambda`
- `euler_steps`
