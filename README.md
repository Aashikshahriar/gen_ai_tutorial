# VAE from Scratch (PyTorch, Apple Silicon MPS)

Implements the ELBO objective from **Theorem 1.4** directly:

```
ELBO_{phi,theta}(x) = - (1/M) sum_m ||x - f_theta(mu_phi(x) + sigma_phi(x) eps^(m))||^2 / (2 sigma_dec^2)
                       + (1/2)( sigma_phi(x)^2 d - d + ||mu_phi(x)||^2 - 2 d log sigma_phi(x) )
```

Key modeling choice: `sigma_phi(x)` is a **scalar** per input (isotropic covariance), matching
the `d * sigma^2` term in the formula — not a per-dimension vector like the more common
diagonal-covariance VAE.

## Directory layout

```
gen_ai/
├── setup.sh              # creates venv + installs requirements
└── vae/
    ├── requirements.txt
    ├── model.py           # Encoder / Decoder / VAE (reparameterization trick)
    ├── losses.py          # ELBO loss, split into recon_term + kl_term
    ├── utils.py           # plotting: loss curves, reconstructions, generated samples
    ├── train.py           # training loop, MPS/CUDA/CPU auto-detect
    └── outputs/           # created at runtime: plots + checkpoint land here
```

## Setup (run on your Mac, not in this sandbox)

```bash
cd gen_ai
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
```

This creates `gen_ai/.venv`, installs `torch`, `torchvision`, `matplotlib`, `numpy`, `tqdm`.

> Note: this was generated in a sandboxed environment with no network/GPU access, so I
> couldn't actually run `pip install` or a training epoch here. Everything is written and
> internally dimension-checked by hand, but please sanity-check the first run yourself.

## Train

```bash
cd vae
python train.py --epochs 20 --batch_size 128 --latent_dim 16 --M 1 --sigma_dec 0.1
```

MNIST downloads automatically to `./data` on first run. Device is auto-detected in this
priority order: **MPS (Apple Silicon) → CUDA → CPU**.

### Important flag: `--kl_sign`

The theorem as transcribed **adds** the KL term to `-recon_term` to form the ELBO. Working
through the algebra, that second term is exactly `KL(q_phi(z|x) || N(0,I))` — and the
*standard* VAE ELBO **subtracts** KL, it doesn't add it (adding it means training would push
the KL term to grow instead of shrink, which is generally unstable / not a real regularizer
toward the prior).

- `--kl_sign literal` reproduces the formula exactly as printed (`-recon + KL`)
- `--kl_sign standard` (**default**) uses the conventional VAE sign (`-recon - KL`), which is
  what will actually train a well-behaved VAE

If the "literal" version is what your course intends (e.g. if there's a sign already baked
into their definition of the KL-like term that isn't obvious from the transcription), switch
the flag and compare the curves — that's exactly what the separate loss plots are for.

## Outputs

After training, `vae/outputs/` will contain:
- `combined_loss.png` — total loss (`-ELBO`) vs epoch
- `loss_components.png` — reconstruction term and KL term as separate subplots
- `all_components_overlay.png` — all three overlaid for comparing tradeoffs
- `reconstructions.png` — real MNIST digits vs their VAE reconstructions
- `generated_samples.png` — pure generation: samples `z ~ N(0,I)` decoded with no input image
  (this is the "sampling/generation" visualization, distinct from reconstruction)
- `vae_mnist.pt` — trained model weights

## Tuning notes

- `latent_dim`: 16 is a reasonable default for MNIST; try 2 if you want to visualize the
  latent space directly as a scatter plot.
- `sigma_dec`: controls the reconstruction/KL tradeoff — smaller values weight
  reconstruction more heavily (sharper but potentially more overfit-to-pixels reconstructions).
- `M`: number of Monte Carlo samples per input in the ELBO estimate. `M=1` is standard and
  usually sufficient; increase for a lower-variance gradient estimate at the cost of compute.