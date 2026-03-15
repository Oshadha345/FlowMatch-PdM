# 04 - Script Reference

This file is the CLI reference for the current repository state.

Phase ownership:
- Phase 0 and Phase 1 and augmented downstream retraining: `train_classifier.py`
- Phase 2 and Phase 3 generator work: `train_generator.py`
- Standalone re-evaluation: `run_evaluation.py`
- Legacy compatibility wrapper: `train_classifier_aug.py`
- End-to-end proof-of-concept: `run_all.sh`

## `train_classifier.py`

Purpose:
- train a baseline classifier/regressor
- train a classically augmented classifier
- train a generator-augmented classifier
- automatically evaluate the trained classifier at the end

Exact CLI:

```bash
python train_classifier.py \
  --track <engine_rul|bearing_rul|bearing_fault> \
  --dataset <DATASET> \
  --model <baseline|LSTMRegressor|CNN1DClassifier> \
  [--run_id <OUTPUT_RUN_ID>] \
  [--aug <none|noise|smote>] \
  [--gen_model <TimeVAE|TimeGAN|DiffusionTS|TimeFlow|COTGAN|FaultDiffusion|FlowMatch>] \
  [--source_run_id <GENERATOR_RUN_ID>] \
  [--gen_ablation <none|no_prior|no_tccm|no_lap>] \
  [--use_wandb] \
  [--config configs/default_config.yaml]
```

Notes:
- `baseline` resolves to `LSTMRegressor` for RUL tracks
- `baseline` resolves to `CNN1DClassifier` for `bearing_fault`
- `--aug noise|smote` selects classical augmentation
- `--gen_model` plus `--source_run_id` selects synthetic augmentation
- writes classifier metrics to `evaluation_results/classifier_metrics.json`
- writes plots to `evaluation_results/classifier_confusion_matrix.png` or `evaluation_results/classifier_regression_diagnostics.png`

## `train_classifier_aug.py`

Purpose:
- compatibility wrapper for older commands

Notes:
- use `train_classifier.py` for new runs
- legacy `--run_id` is remapped to `--source_run_id`

## `train_generator.py`

Purpose:
- train a generator on the minority subset
- automatically evaluate it after training
- support FlowMatch-PdM ablations
- support direct W&B sweep parameter overrides

Exact CLI:

```bash
python train_generator.py \
  --track <engine_rul|bearing_rul|bearing_fault> \
  --dataset <DATASET> \
  --model <TimeVAE|TimeGAN|DiffusionTS|TimeFlow|COTGAN|FaultDiffusion|FlowMatch> \
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
- `FlowMatch` is the CLI name for FlowMatch-PdM
- FlowMatch ablations are routed into their own model folders via `FlowMatch_ablation_<name>`
- generator evaluation writes `generator_datas/*.npy` and `evaluation_results/metrics.json`
- when W&B is enabled, post-training generator metrics are logged with the `evaluation/` prefix

## `run_evaluation.py`

Purpose:
- re-evaluate an existing classifier run
- re-evaluate an existing generator run

Exact CLI:

```bash
python run_evaluation.py \
  --eval_mode <classifier|generator> \
  --track <engine_rul|bearing_rul|bearing_fault> \
  --dataset <DATASET> \
  --model <MODEL_NAME> \
  [--run_id <RUN_ID>] \
  [--aug <none|noise|smote>] \
  [--source_gen_model <GENERATOR_MODEL>] \
  [--source_run_id <GENERATOR_RUN_ID>] \
  [--ablation <none|no_prior|no_tccm|no_lap>] \
  [--gen_ablation <none|no_prior|no_tccm|no_lap>] \
  [--config configs/default_config.yaml]
```

Generator examples:

```bash
python run_evaluation.py --eval_mode generator --track engine_rul --dataset CMAPSS --model FlowMatch
python run_evaluation.py --eval_mode generator --track engine_rul --dataset CMAPSS --model FlowMatch --ablation no_prior --run_id run_20260316_120000
```

Classifier examples:

```bash
python run_evaluation.py --eval_mode classifier --track engine_rul --dataset CMAPSS --model baseline
python run_evaluation.py --eval_mode classifier --track engine_rul --dataset CMAPSS --model baseline --source_gen_model FlowMatch --source_run_id run_20260316_120000
```

Notes:
- classifier mode writes `classifier_metrics.json` and classifier plots
- generator mode writes `metrics.json`, PCA/t-SNE, KDE, and synthetic arrays
- TSTR is part of generator mode and is saved as `predictive_score_mae`

## `run_all.sh`

Purpose:
- end-to-end CMAPSS proof-of-concept pipeline

Exact usage:

```bash
./run_all.sh
```

Environment overrides:

```bash
PYTHON_BIN=/home/buddhiw/miniconda3/envs/flowmatch_pdm/bin/python ./run_all.sh
GEN_MODELS="FlowMatch COTGAN FaultDiffusion" ./run_all.sh
DATASET=CMAPSS TRACK=engine_rul ./run_all.sh
```

Behavior:
- trains the baseline classifier
- trains each requested generator
- relies on generator auto-evaluation
- retrains the classifier on generator-augmented data
- relies on classifier auto-evaluation

## `configs/sweep_flowmatch_cmapss.yaml`

Purpose:
- W&B Bayesian sweep for FlowMatch-PdM on CMAPSS

Exact usage:

```bash
wandb sweep configs/sweep_flowmatch_cmapss.yaml
wandb agent <SWEEP_ID>
```

Optimized parameters:
- `lr`
- `batch_size`
- `epochs`
- `euler_steps`
- `mamba_d_model`
- `mamba_d_state`
- `tccm_lambda`
- `lap_threshold`

Primary sweep objective:
- `evaluation/ftsd`
