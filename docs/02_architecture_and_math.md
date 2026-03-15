# 02 - Architecture And Math

This file describes the mathematics that are actually implemented in the current repository.
It matches the phase layout documented in `docs/01_experiment_plan.md` and the CLIs documented in `docs/04_script_reference.md`.

## 1. Dataset Tensor Contract

All training code assumes:

```text
X: [batch, window, features]
y:
  float32 for RUL regression
  int64 for classification
```

This is enforced in `datasets/rul_data_loader.py` for `rul-datasets` readers and is already native for the `.npy` classification loaders.

## 2. FlowMatch-PdM

Implemented in `flowmatchPdM/flowmatch_pdm.py`.

Naming contract:
- CLI model name: `FlowMatch`
- implementation class: `FlowMatchPdM`
- config block: `generative.flowmatch_pdm`
- ablation flags: `no_prior`, `no_tccm`, `no_lap`

### 2.1 Training Path

For a real sequence `x1` and condition `c`:

```text
x0 = harmonic_prior(c)
xt = (1 - t) * x0 + t * x1
ut = x1 - x0
vt = f_theta(t, xt, c)
loss = MSE(vt, ut) + TCCM(vt, c)
```

Where:
- `harmonic_prior(c)` is the condition-driven base distribution
- `f_theta` is the Mamba-based vector field
- `TCCM` is the contraction-style penalty from `flowmatchPdM/model/tccm_loss.py`

### 2.2 Model Structure

The implemented forward path is:

```text
[x, t] -> Linear(input_dim + 1, d_model)
      -> 3 x BidirectionalMambaBlock
      -> Linear(d_model, input_dim)
```

### 2.3 Sampling Path

Generation starts from the harmonic prior and solves the ODE with Euler integration over `euler_steps`:

```text
dx / dt = f_theta(t, x, c)
```

## 3. Baseline Generators

Implemented in `src/baselines.py`.

## 3.1 TimeVAE

Encoder-decoder recurrent VAE:

```text
q(z | x) = N(mu(x), sigma(x))
loss = MSE(x_hat, x) + beta_kl * KL(q(z | x) || N(0, I))
```

The current implementation uses:
- 2-layer LSTM encoder
- linear `mu` and `logvar` heads
- 2-layer LSTM decoder

## 3.2 TimeGAN

The current implementation is a practical recurrent GAN baseline with feature matching:

```text
d_loss = 0.5 * [BCE(D(x_real), 0.9) + BCE(D(x_fake), 0.0)]
g_loss = BCE(D(x_fake), 1.0) + lambda_fm * (MSE(mu_fake, mu_real) + MSE(std_fake, std_real))
```

Where `mu_*` and `std_*` are per-feature sequence statistics.

## 3.3 DiffusionTS

DDPM objective with a 1D U-Net denoiser:

```text
xt = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
loss = MSE(eps_theta(xt, t), eps)
```

Sampling uses the standard reverse DDPM mean update with learned noise prediction.

## 3.4 FaultDiffusion

Bearing-oriented diffusion model with:
- 1D U-Net backbone
- attention in the bottleneck
- adapter blocks for high-frequency fault structure

Training objective:

```text
denoise_loss = 0.5 * [MSE(eps_theta(x_t1, t1), eps1) + MSE(eps_theta(x_t2, t2), eps2)]
diversity_bonus = mean_pairwise_distance(x0_hat) / mean_pairwise_distance(x0)
loss = denoise_loss - diversity_weight * diversity_bonus
```

This is the exact current repo behavior.

## 3.5 TimeFlow

Standard flow matching baseline:

```text
x0 ~ N(0, I)
xt = (1 - t) * x0 + t * x1
ut = x1 - x0
loss = MSE(v_theta(t, xt), ut)
```

Generation uses explicit Euler integration.

## 3.6 COTGAN

The current implementation uses:
- LSTM generator
- causal critic with GRU backbone
- mixed Sinkhorn divergence
- martingale regularization

The implemented pairwise causal cost is:

```text
C(x, y) =
  mean(||x - y||^2)
  + mean(||h(x) - h(y)||^2)
  + causal_weight * causal_term
```

Where:
- `h(.)` is the critic feature projection
- `causal_term` is built from critic martingale increments and prefix features

The implemented mixed Sinkhorn divergence is:

```text
D_mix(x, y) = S_eps(C_xy) - 0.5 * S_eps(C_xx) - 0.5 * S_eps(C_yy)
```

With current optimization:

```text
critic_loss = -D_mix(real, fake) + martingale_weight * martingale_penalty
generator_loss = D_mix(real, fake)
```

This is the exact repo math, not a claim of full paper reproduction.

## 4. Classifiers

Implemented in `src/classifier.py`.

## 4.1 LSTMRegressor

For RUL tasks:

```text
h = LSTM(x)
y_hat = MLP(h_last)
loss = MSE(y_hat, y)
```

Test metrics logged:
- `test_mse`
- `test_rmse`
- `test_mae`
- `test_r2`
- `test_mape`
- `test_smape`

## 4.2 CNN1DClassifier

For classification tasks:

```text
x -> 5 Conv1D blocks -> GAP -> MLP -> logits
loss = CrossEntropy(logits, y)
```

The model now exposes `extract_features(x)` and supports multi-channel inputs such as DEMADICS.

## 5. Minority Subset Logic

### 5.1 RUL

Implemented in `datasets/rul_data_loader.py`.

For train split targets `y_train`:

```text
threshold = 0.2 * max_rul
minority = { i : y_train[i] <= threshold }
```

Special case:
- `CMAPSS` uses clipped `max_rul = 125`
- other RUL datasets use the train-split maximum after reader preprocessing

### 5.2 Classification

Implemented in `datasets/cwru_data_loader.py`, `datasets/paderborn_data_loader.py`, and `datasets/demadics_data_loader.py`.

Minority extraction is:

```text
minority_labels = argmin_class_count(train_labels)
minority_dataset = all train windows with labels in minority_labels
```

## 6. Evaluation Metrics

Implemented in `src/evaluation.py`.

There are now two evaluation modes:
- classifier / regressor evaluation
- generator / synthetic-data evaluation

## 6.1 FTSD

Deep features are extracted from the Phase 0 baseline model if provided. Otherwise flattened windows are used.

```text
FTSD = ||mu_r - mu_s||^2 + Tr(Sigma_r + Sigma_s - 2 * sqrt(Sigma_r Sigma_s))
```

## 6.2 MMD

Uses an RBF kernel with median-distance bandwidth:

```text
MMD = E[k(x, x')] + E[k(y, y')] - 2E[k(x, y)]
```

The implementation clamps the final reported value to `>= 0`.

## 6.3 Discriminative Score

A post-hoc GRU is trained to classify real vs synthetic sequences:

```text
score = |0.5 - accuracy|
```

Lower is better.

## 6.4 Predictive Score

Train on synthetic, test on real:

```text
input  = x[:, :-1, :]
target = x[:, 1:, :]
score  = mean_absolute_error(target_real, predictor(input_real))
```

Lower is better.

## 6.5 Visualizations

The evaluator writes:
- `projection_pca_tsne.png`
- `marginal_kde.png`

These are generated from the same arrays used for metric evaluation.
