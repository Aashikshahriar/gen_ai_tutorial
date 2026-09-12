# Generative Models from Scratch (PyTorch, Apple Silicon MPS)

Small, self-contained implementations of generative-modeling theory, each in its own
subfolder with its own `train.py`. Shared Python environment at `./venv`.

```
gen_AI/
├── venv/                   # shared virtualenv (torch, torchvision, matplotlib, numpy, tqdm)
├── vae/                    # Variational Autoencoder (Theorem 1.4's ELBO)
├── diffusion_forward/      # forward-diffusion visualizations (no training) — Example 2.1 / Fig 2.5, Fig 2.7
├── ddpm/                   # DDPM, x0-prediction denoiser
└── ddpm_eps_pred/          # DDPM, noise (epsilon)-prediction denoiser
```

## Setup

```bash
cd gen_AI
python3 -m venv venv
source venv/bin/activate
pip install -r vae/requirements.txt   # same deps for every subfolder: torch, torchvision, matplotlib, numpy, tqdm
```

Device is auto-detected in each `train.py` in this priority order: **MPS (Apple Silicon) → CUDA → CPU**.

MNIST is downloaded by hand (no `torchvision.datasets`, see `mnist_loader.py` in each subfolder that
uses it) to a local `./data/` on first run, from `storage.googleapis.com/cvdf-datasets/mnist/`
(the original `yann.lecun.com` / S3 mirrors are dead).

---

## `vae/` — Variational Autoencoder

Implements the ELBO objective from VAE paper directly:

```
ELBO_{phi,theta}(x) = - (1/M) sum_m ||x - f_theta(mu_phi(x) + sigma_phi(x) eps^(m))||^2 / (2 sigma_dec^2)
                       + (1/2)( sigma_phi(x)^2 d - d + ||mu_phi(x)||^2 - 2 d log sigma_phi(x) )
```

Key modeling choice: `sigma_phi(x)` is a **scalar** per input (isotropic covariance), matching
the `d * sigma^2` term in the formula — not a per-dimension vector like the more common
diagonal-covariance VAE.

Layout: `model.py` (Encoder/Decoder/VAE), `losses.py` (ELBO split into recon_term + kl_term),
`utils.py` (plotting), `train.py`, `mnist_loader.py`.

```bash
cd vae
python train.py --epochs 20 --batch_size 128 --latent_dim 16 --M 1 --sigma_dec 0.1
```

### `--kl_sign`

The theorem as transcribed **adds** the KL term to `-recon_term` to form the ELBO. Working
through the algebra, that second term is exactly `KL(q_phi(z|x) || N(0,I))` — and the
*standard* VAE ELBO **subtracts** KL, it doesn't add it (adding it means training would push
the KL term to grow instead of shrink, which is generally unstable / not a real regularizer
toward the prior).

- `--kl_sign literal` reproduces the formula exactly as printed (`-recon + KL`)
- `--kl_sign standard` (**default**) uses the conventional VAE sign (`-recon - KL`), which is
  what will actually train a well-behaved VAE

### Outputs (`vae/outputs/`)

- `combined_loss.pdf` — total loss (`-ELBO`) vs epoch
- `loss_components.pdf` — reconstruction term and KL term as separate subplots
- `all_loss_components.pdf` — all three overlaid for comparing tradeoffs
- `reconstructions.pdf` — real MNIST digits vs their VAE reconstructions
- `generated_samples.pdf` — pure generation: samples `z ~ N(0,I)` decoded with no input image
- `vae_mnist.pt` — trained model weights

### Tuning notes

- `latent_dim`: 16 is a reasonable default for MNIST; try 2 if you want to visualize the
  latent space directly as a scatter plot.
- `sigma_dec`: controls the reconstruction/KL tradeoff — smaller values weight
  reconstruction more heavily (sharper but potentially more overfit-to-pixels reconstructions).
- `M`: number of Monte Carlo samples per input in the ELBO estimate. `M=1` is standard and
  usually sufficient; increase for a lower-variance gradient estimate at the cost of compute.

---

## `diffusion_forward/` — forward-diffusion visualizations

No training — closed-form replications of the forward-diffusion figures for a 1D Gaussian
mixture `x0 ~ pi_1 N(mu_1, sigma_1^2) + pi_2 N(mu_2, sigma_2^2)` (Example 2.1).

- `gmm_forward_diffusion.py` — replicates **Figure 2.5**: recursively applies the eq. (2.2)
  update to the per-component mean/variance and plots `p_t(x)` at `t = 1, 5, 10, 30, 40, 50, 100, 200`.
  Output: `forward_diffusion.png`.
- `trajectories.py` — replicates **Figure 2.7**: closed-form heatmap of `p_t(x)` from eq. (2.13)
  (`bar_alpha_t = alpha^t` under constant `alpha_t`), with a few simulated sample paths
  `x_t = sqrt(alpha) x_{t-1} + sqrt(1-alpha) eps` overlaid on top. Output: `trajectories.png`.

```bash
cd diffusion_forward
python gmm_forward_diffusion.py
python trajectories.py
```

---

## `ddpm/` — DDPM, x0-prediction

Denoiser `x_theta(x_t, t)` predicts the **clean image `x0` directly**, trained per the DDPM
training algorithm:

```
t ~ Uniform[1, T]
x_t = sqrt(bar_alpha_t) x0 + sqrt(1 - bar_alpha_t) eps,   eps ~ N(0, I)
loss = || x_theta(x_t, t) - x0 ||^2
```

Sampling (inference) starts from `x_T ~ N(0, I)` and recursively applies the DDPM reverse
posterior `x_{t-1} ~ N(mu_q(x_t, x0_hat), sigma_q(t)^2 I)` down to `x_0`.

Layout: `diffusion.py` (beta/alpha/bar_alpha schedule, forward `q_sample`, reverse posterior
mean/variance), `model.py` (small U-Net with sinusoidal time embedding, `Denoiser`),
`losses.py`, `train.py`, `utils.py`, `mnist_loader.py`.

```bash
cd ddpm
python train.py --epochs 20 --batch_size 128 --T 300
```

### Outputs (`ddpm/outputs/`)

- `loss_curve.pdf` — denoising MSE vs epoch
- `generated_samples.pdf` — 16 samples generated via reverse diffusion from white noise
- `reverse_trajectory.pdf` — one sample's path from `x_T` down to `x_0`, snapshotted every ~10% of `T`
- `ddpm_mnist.pt` — trained model weights

---

## `ddpm_eps_pred/` — DDPM, noise-prediction (epsilon)

Same U-Net/schedule/training loop as `ddpm/`, but the network `eps_theta(x_t, t)` predicts the
**noise `eps`** that was added to `x0` (the parameterization used in Ho et al. 2020's original
DDPM paper), not `x0` itself:

```
eps ~ N(0, I),  x_t = sqrt(bar_alpha_t) x0 + sqrt(1 - bar_alpha_t) eps
loss = || eps_theta(x_t, t) - eps ||^2
```

At each reverse-sampling step, `eps_hat` is converted back to an `x0_hat` estimate via
`x0_hat = (x_t - sqrt(1 - bar_alpha_t) eps_hat) / sqrt(bar_alpha_t)`, then the same DDPM
posterior update as `ddpm/` is applied. Mathematically equivalent to the x0-prediction loss up
to a per-`t` reweighting, but generally trains more stably.

```bash
cd ddpm_eps_pred
python train.py --epochs 20 --batch_size 128 --T 300
```

Outputs land in `ddpm_eps_pred/outputs/` (same set as `ddpm/`, checkpoint named `ddpm_eps_mnist.pt`).
