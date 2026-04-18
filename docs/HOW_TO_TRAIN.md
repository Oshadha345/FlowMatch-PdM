# FlowMatch-PdM RUL Pivot Training Guide

## Environment

```bash
conda env create -f environment.yml
conda activate flowmatch_pdm
pip install -r requirements.txt
```

If `mamba-ssm` or `causal-conv1d` fail to build locally:

```bash
pip install whl/causal_conv1d-1.5.0.post8+cu11torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install whl/mamba_ssm-2.2.4+cu11torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## Active Campaign

Only the following scope is active for the hackathon run:

- `bearing_rul / FEMTO`
- `bearing_rul / XJTU-SY`
- downstream regressor: `MambaRULRegressor`
- classical augmentation: `noise`, `smote`
- generators: `COTGAN`, `FaultDiffusion`, `DiffusionTS`, `TimeFlow`, `FlowMatch`
- ablations: `FlowMatch --ablation no_tccm`, `FlowMatch --ablation no_prior`, `FlowMatch --ablation no_lap`

## One-Command Execution

```bash
bash final_execution.sh
```

This runner:

- trains the primary FlowMatch runs first
- trains the FlowMatch downstream Mamba TSTR runs immediately after
- trains the classical RUL Mamba augmentations next
- trains the heavy generator baselines after the FlowMatch priority block
- trains downstream Mamba regressors on synthetic data
- runs the FEMTO-only FlowMatch ablations at the end
- triggers the FEMTO W&B sweep command in Step 4

## Non-FlowMatch Background Queue

Use this when `FlowMatch` is being run manually on another GPU and you want the rest of the campaign to proceed sequentially:

```bash
CUDA_VISIBLE_DEVICES=0 RUN_TAG=pivot_rul_gpu0_background_20260414 python3 orchestrate.py
```

This path now runs only:

- classical `noise` and `smote` Mamba baselines
- `COTGAN`, `FaultDiffusion`, `TimeFlow`, and `DiffusionTS`
- downstream Mamba TSTR for those non-FlowMatch generators

## Tmux Execution

```bash
bash launch.sh
tmux attach -t flowmatch
```

## Manual Commands

### Classical Augmentation

```bash
python train_classifier.py --track bearing_rul --dataset FEMTO --eval_model mamba --aug noise --run_id pivot_rul_mamba_20260414_femto_mamba_noise
python train_classifier.py --track bearing_rul --dataset FEMTO --eval_model mamba --aug smote --run_id pivot_rul_mamba_20260414_femto_mamba_smote
python train_classifier.py --track bearing_rul --dataset XJTU-SY --eval_model mamba --aug noise --run_id pivot_rul_mamba_20260414_xjtu_sy_mamba_noise
python train_classifier.py --track bearing_rul --dataset XJTU-SY --eval_model mamba --aug smote --run_id pivot_rul_mamba_20260414_xjtu_sy_mamba_smote
```

### Missing Generator Runs

```bash
python train_generator.py --track bearing_rul --dataset FEMTO --model FlowMatch --run_id pivot_rul_mamba_20260414_femto_flowmatch
python train_generator.py --track bearing_rul --dataset XJTU-SY --model FlowMatch --run_id pivot_rul_mamba_20260414_xjtu_sy_flowmatch
python train_generator.py --track bearing_rul --dataset FEMTO --model COTGAN --run_id pivot_rul_mamba_20260414_femto_cotgan
python train_generator.py --track bearing_rul --dataset FEMTO --model FaultDiffusion --run_id pivot_rul_mamba_20260414_femto_faultdiffusion
python train_generator.py --track bearing_rul --dataset FEMTO --model TimeFlow --run_id pivot_rul_mamba_20260414_femto_timeflow
python train_generator.py --track bearing_rul --dataset FEMTO --model DiffusionTS --run_id pivot_rul_mamba_20260414_femto_diffusionts
python train_generator.py --track bearing_rul --dataset XJTU-SY --model COTGAN --run_id pivot_rul_mamba_20260414_xjtu_sy_cotgan
python train_generator.py --track bearing_rul --dataset XJTU-SY --model FaultDiffusion --run_id pivot_rul_mamba_20260414_xjtu_sy_faultdiffusion
python train_generator.py --track bearing_rul --dataset XJTU-SY --model TimeFlow --run_id pivot_rul_mamba_20260414_xjtu_sy_timeflow
python train_generator.py --track bearing_rul --dataset XJTU-SY --model DiffusionTS --run_id pivot_rul_mamba_20260414_xjtu_sy_diffusionts
```

### Downstream Synthetic TSTR

```bash
python train_classifier_aug.py --track bearing_rul --dataset FEMTO --eval_model mamba --gen_model FlowMatch --source_run_id pivot_rul_mamba_20260414_femto_flowmatch --output_run_id pivot_rul_mamba_20260414_femto_flowmatch_mamba_tstr
python train_classifier_aug.py --track bearing_rul --dataset XJTU-SY --eval_model mamba --gen_model FlowMatch --source_run_id pivot_rul_mamba_20260414_xjtu_sy_flowmatch --output_run_id pivot_rul_mamba_20260414_xjtu_sy_flowmatch_mamba_tstr
python train_classifier_aug.py --track bearing_rul --dataset FEMTO --eval_model mamba --gen_model COTGAN --source_run_id pivot_rul_mamba_20260414_femto_cotgan --output_run_id pivot_rul_mamba_20260414_femto_cotgan_mamba_tstr
python train_classifier_aug.py --track bearing_rul --dataset FEMTO --eval_model mamba --gen_model FaultDiffusion --source_run_id pivot_rul_mamba_20260414_femto_faultdiffusion --output_run_id pivot_rul_mamba_20260414_femto_faultdiffusion_mamba_tstr
python train_classifier_aug.py --track bearing_rul --dataset FEMTO --eval_model mamba --gen_model TimeFlow --source_run_id pivot_rul_mamba_20260414_femto_timeflow --output_run_id pivot_rul_mamba_20260414_femto_timeflow_mamba_tstr
python train_classifier_aug.py --track bearing_rul --dataset FEMTO --eval_model mamba --gen_model DiffusionTS --source_run_id pivot_rul_mamba_20260414_femto_diffusionts --output_run_id pivot_rul_mamba_20260414_femto_diffusionts_mamba_tstr
python train_classifier_aug.py --track bearing_rul --dataset XJTU-SY --eval_model mamba --gen_model COTGAN --source_run_id pivot_rul_mamba_20260414_xjtu_sy_cotgan --output_run_id pivot_rul_mamba_20260414_xjtu_sy_cotgan_mamba_tstr
python train_classifier_aug.py --track bearing_rul --dataset XJTU-SY --eval_model mamba --gen_model FaultDiffusion --source_run_id pivot_rul_mamba_20260414_xjtu_sy_faultdiffusion --output_run_id pivot_rul_mamba_20260414_xjtu_sy_faultdiffusion_mamba_tstr
python train_classifier_aug.py --track bearing_rul --dataset XJTU-SY --eval_model mamba --gen_model TimeFlow --source_run_id pivot_rul_mamba_20260414_xjtu_sy_timeflow --output_run_id pivot_rul_mamba_20260414_xjtu_sy_timeflow_mamba_tstr
python train_classifier_aug.py --track bearing_rul --dataset XJTU-SY --eval_model mamba --gen_model DiffusionTS --source_run_id pivot_rul_mamba_20260414_xjtu_sy_diffusionts --output_run_id pivot_rul_mamba_20260414_xjtu_sy_diffusionts_mamba_tstr
```

### FlowMatch Ablations

```bash
python train_generator.py --track bearing_rul --dataset FEMTO --model FlowMatch --ablation no_tccm --run_id pivot_rul_20260414_femto_flowmatch_no_tccm
python train_classifier_aug.py --track bearing_rul --dataset FEMTO --eval_model mamba --gen_model FlowMatch --gen_ablation no_tccm --source_run_id pivot_rul_20260414_femto_flowmatch_no_tccm --output_run_id pivot_rul_20260414_femto_flowmatch_no_tccm_mamba_tstr
python train_generator.py --track bearing_rul --dataset FEMTO --model FlowMatch --ablation no_prior --run_id pivot_rul_20260414_femto_flowmatch_no_prior
python train_classifier_aug.py --track bearing_rul --dataset FEMTO --eval_model mamba --gen_model FlowMatch --gen_ablation no_prior --source_run_id pivot_rul_20260414_femto_flowmatch_no_prior --output_run_id pivot_rul_20260414_femto_flowmatch_no_prior_mamba_tstr
python train_generator.py --track bearing_rul --dataset FEMTO --model FlowMatch --ablation no_lap --run_id pivot_rul_20260414_femto_flowmatch_no_lap
python train_classifier_aug.py --track bearing_rul --dataset FEMTO --eval_model mamba --gen_model FlowMatch --gen_ablation no_lap --source_run_id pivot_rul_20260414_femto_flowmatch_no_lap --output_run_id pivot_rul_20260414_femto_flowmatch_no_lap_mamba_tstr
```

### W&B Sweep

```bash
wandb sweep configs/sweep_flowmatch_femto.yaml
wandb agent <SWEEP_ID>
```

## Metrics

For RUL runs, `evaluation_results/classifier_metrics.json` includes:

- `rmse`
- `mae`
- `r2`
- `mse`
- `explained_variance`
- `median_ae`
- `max_error`
