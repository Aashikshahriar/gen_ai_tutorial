"""
Replicates Example 4.7 / Figure 4.6: the reverse diffusion SDE, discretized
exactly as in the DDPM-consistent update recovered from the VP-SDE:

    x_{i-1} = (1 / sqrt(1 - beta_i)) * [ x_i + (beta_i / 2) * grad_x log p_i(x_i) ] + sqrt(beta_i) * z_i,
    z_i ~ N(0, I)

Run for i = T, T-1, ..., 1, starting from x_T ~ N(0, I) (the forward process's
near-Gaussian marginal at t=1), this recovers samples from the *original*
Gaussian mixture p_0(x) at i=0 -- i.e. reverse diffusion turns pure noise back
into the data distribution.

Since p_0 is the same known 1D Gaussian mixture used throughout (Example 2.1),
every intermediate marginal p_i(x) is available in closed form (same math as
diffusion_forward/gmm_forward_diffusion.py and sde_diffusion/forward_sde.py),
so grad_x log p_i(x) -- the TRUE score, not a learned approximation -- can be
computed analytically at every step.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- mixture / DDPM-consistent schedule (same mixture as Example 2.1) ---
PI = (0.3, 0.7)
MU0 = (-2.0, 2.0)
SIGMA0 = (0.2, 1.0)

T = 1000
BETA_1, BETA_T = 1e-4, 0.02

N_TRAJECTORIES = 5
X_RANGE = (-5, 5)
N_X = 300

rng = np.random.default_rng(0)

betas = np.linspace(BETA_1, BETA_T, T)          # beta_i, i = 1..T (index i-1)
alphas = 1.0 - betas
bar_alphas = np.cumprod(alphas)                 # bar_alpha_i, i = 1..T (index i-1)
bar_alphas_full = np.concatenate([[1.0], bar_alphas])  # index i = 0..T, bar_alpha_0 := 1


def marginal_pdf(x, bar_alpha):
    density = np.zeros_like(x)
    for pi_k, mu_k, sigma_k in zip(PI, MU0, SIGMA0):
        mean = np.sqrt(bar_alpha) * mu_k
        var = (1 - bar_alpha) + bar_alpha * sigma_k ** 2
        density += pi_k * np.exp(-0.5 * (x - mean) ** 2 / var) / np.sqrt(2 * np.pi * var)
    return density


def score(x, bar_alpha):
    """True grad_x log p_i(x) for the mixture marginal at bar_alpha_i."""
    comp_pdfs, comp_scores = [], []
    for pi_k, mu_k, sigma_k in zip(PI, MU0, SIGMA0):
        mean = np.sqrt(bar_alpha) * mu_k
        var = (1 - bar_alpha) + bar_alpha * sigma_k ** 2
        comp_pdfs.append(pi_k * np.exp(-0.5 * (x - mean) ** 2 / var) / np.sqrt(2 * np.pi * var))
        comp_scores.append(-(x - mean) / var)
    total = sum(comp_pdfs) + 1e-12
    weighted = sum(w * s for w, s in zip(comp_pdfs, comp_scores))
    return weighted / total


def reverse_sample(n_particles):
    x = rng.normal(size=n_particles)  # x_T ~ N(0, I)
    traj = np.zeros((T + 1, n_particles))
    traj[0] = x  # iteration k=0 <-> i=T

    for k in range(1, T + 1):
        i = T - k + 1  # current index before this update, i = T, T-1, ..., 1
        beta_i = betas[i - 1]
        ba_i = bar_alphas_full[i]
        s = score(x, ba_i)
        z = rng.normal(size=n_particles)
        x = (x + (beta_i / 2) * s) / np.sqrt(1 - beta_i) + np.sqrt(beta_i) * z
        traj[k] = x

    return traj


def run():
    traj = reverse_sample(N_TRAJECTORIES)

    ks = np.arange(0, T + 1, 5)  # subsample iterations for the heatmap
    xs = np.linspace(*X_RANGE, N_X)
    heatmap = np.array([marginal_pdf(xs, bar_alphas_full[T - k]) for k in ks])

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.pcolormesh(ks, xs, heatmap.T, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="$p_i(x)$")

    colors = ["white", "silver", "dimgray", "black", "darkgray"]
    for j in range(N_TRAJECTORIES):
        ax.plot(np.arange(T + 1), traj[:, j], color=colors[j % len(colors)], linewidth=1)

    ax.invert_yaxis()
    ax.set_ylim(5, -5)
    ax.set_xlabel("reverse iteration")
    ax.set_ylabel("x")
    ax.set_title("Reverse diffusion SDE: single Gaussian $\\to$ Gaussian mixture")
    fig.tight_layout()

    out_path = "reverse_sde.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    run()
