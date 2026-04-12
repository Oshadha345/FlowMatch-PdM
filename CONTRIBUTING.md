# Contributing to FlowMatch-PdM

## Adding a New Generator Baseline

Follow these steps to integrate a new generative model into the experiment pipeline.

### 1. Implement the Model Class

Add your model class to `src/baselines.py`. It must:

- Extend `pytorch_lightning.LightningModule`
- Accept `input_dim: int` and `window_size: int` as constructor arguments
- Implement `training_step(self, batch, batch_idx)` that logs `train_loss`
- Implement `configure_optimizers(self)`
- Implement `generate(self, num_samples: int, conditions=None) -> torch.Tensor`
  returning a tensor of shape `[num_samples, window_size, input_dim]`

```python
class MyNewGenerator(pl.LightningModule):
    def __init__(self, input_dim: int, window_size: int, lr: float = 1e-3, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        # ... build your model layers ...

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        # ... compute loss ...
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    @torch.no_grad()
    def generate(self, num_samples: int, conditions=None) -> torch.Tensor:
        # Return shape: [num_samples, window_size, input_dim]
        ...
```

### 2. Register in the Model Builder

In `train_generator.py`, add your model to two places:

**a) Generator choices and config map (top of file):**

```python
GENERATOR_CHOICES = ["TimeVAE", "TimeGAN", ..., "MyNewGenerator"]
GENERATOR_CONFIG_MAP = {
    ...
    "MyNewGenerator": "my_new_generator",
}
```

**b) The `_build_generator()` function:**

```python
if model_name == "MyNewGenerator":
    return MyNewGenerator(
        input_dim=input_dim,
        window_size=window_size,
        lr=float(model_cfg["lr"]),
        # ... other params from model_cfg ...
    )
```

### 3. Add Configuration Block

Add a config section to `configs/default_config.yaml` under `generative:`:

```yaml
generative:
  # ... existing models ...
  my_new_generator:
    lr: 0.001
    epochs: 200
    batch_size: 64
    # ... model-specific hyperparameters ...
```

### 4. Add to Orchestrator

In `orchestrate.py`, add your model name to the `GENERATORS` list:

```python
GENERATORS = [
    "TimeVAE", "TimeGAN", "COTGAN", "FaultDiffusion",
    "DiffusionTS", "TimeFlow", "FlowMatch", "MyNewGenerator",
]
```

The orchestrator automatically handles training, evaluation, and result collection for all models in this list.

### 5. Update pipeline_state.json

Add entries for the new model in the `phase2` section. You can use `scripts/resume_from.py` to manage state, or manually add entries:

```json
"MyNewGenerator__engine_rul__CMAPSS": {"gen_status": "pending", "gen_run_id": null, "clf_status": "pending", "clf_run_id": null},
"MyNewGenerator__bearing_fault__CWRU": {"gen_status": "pending", "gen_run_id": null, "clf_status": "pending", "clf_run_id": null},
"MyNewGenerator__bearing_fault__DEMADICS": {"gen_status": "pending", "gen_run_id": null, "clf_status": "pending", "clf_run_id": null}
```

### 6. Re-run the Pipeline

```bash
bash launch.sh
```

The orchestrator will automatically pick up the new model and train it on all primary datasets.

## Code Style

- Python 3.10+
- Follow existing patterns in the codebase
- Use type hints for function signatures
- Use `float()` when reading numeric values from YAML configs (PyYAML may parse scientific notation as strings)

## Testing

After adding a model, verify it works with a quick smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python train_generator.py \
  --track bearing_fault --dataset CWRU --model MyNewGenerator \
  --epochs 2 --batch_size 8
```
