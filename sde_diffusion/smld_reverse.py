"""
Replicates Theorem 4.4: the reverse-time SDE for SMLD (VE-SDE),

    dx = -( d[sigma(t)^2]/dt * grad_x log p_t(x) ) dt + sqrt( d[sigma(t)^2]/dt ) dw_bar   (Eq. 4.20)

discretized (annealed Langevin dynamics / NCSN-style ancestral sampler):

    x_{i-1} = x_i + (sigma_i^2 - sigma_{i-1}^2) * grad_x log p_i(x_i) + sqrt(sigma_i^2 - sigma_{i-1}^2) * z_i,
    z_i ~ N(0, I)

run for i = L, L-1, ..., 1, starting from x_L ~ N(0, sigma_max^2 I) (pure noise
at the largest scale). Since the data distribution p_0 is the known 1D
Gaussian mixture from Example 2.1, and the VE-SDE only adds variance (means
and weights untouched, see smld_forward.py), grad_x log p_i(x) -- the TRUE
score, not a learned approximation -- is available in closed form at every
step, exactly as in reverse_sde.py for the VP-SDE/DDPM case.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- mixture / VE-SDE schedule (same mixture as Example 2.1, same schedule as smld_forward.py) ---
PI = (0.3, 0.7)
MU0 = (-2.0, 2.0)
SIGMA0 = (0.2, 1.0)

SIGMA_MIN, SIGMA_MAX = 0.01, 10.0
L = 1000  # number of noise scales / reverse steps

N_TRAJECTORIES = 5
X_RANGE = (-15, 15)
N_X = 300

rng = np.random.default_rng(0)

# sigma_i for i = 0..L (geometric schedule), sigma_0 = SIGMA_MIN (~0), sigma_L = SIGMA_MAX
i_grid = np.arange(L + 1)
sigmas = SIGMA_MIN * (SIGMA_MAX / SIGMA_MIN) ** (i_grid / L)


def marginal_pdf(x, sigma_i):
    var_extra = sigma_i ** 2
    density = np.zeros_like(x)
    for pi_k, mu_k, sigma_k in zip(PI, MU0, SIGMA0):
        var = sigma_k ** 2 + var_extra
        density += pi_k * np.exp(-0.5 * (x - mu_k) ** 2 / var) / np.sqrt(2 * np.pi * var)
    return density


def score(x, sigma_i):
    """True grad_x log p_i(x) for the mixture marginal at noise scale sigma_i."""
    var_extra = sigma_i ** 2
    comp_pdfs, comp_scores = [], []
    for pi_k, mu_k, sigma_k in zip(PI, MU0, SIGMA0):
        var = sigma_k ** 2 + var_extra
        comp_pdfs.append(pi_k * np.exp(-0.5 * (x - mu_k) ** 2 / var) / np.sqrt(2 * np.pi * var))
        comp_scores.append(-(x - mu_k) / var)
    total = sum(comp_pdfs) + 1e-12
    weighted = sum(w * s for w, s in zip(comp_pdfs, comp_scores))
    return weighted / total


def reverse_sample(n_particles):
    x = rng.normal(0, SIGMA_MAX, size=n_particles)  # x_L ~ N(0, sigma_max^2)
    traj = np.zeros((L + 1, n_particles))
    traj[0] = x  # iteration k=0 <-> i=L

    for k in range(1, L + 1):
        i = L - k + 1  # step from sigma_i down to sigma_{i-1}
        sigma_i = sigmas[i]
        sigma_im1 = sigmas[i - 1]
        step_var = sigma_i ** 2 - sigma_im1 ** 2
        s = score(x, sigma_i)
        z = rng.normal(size=n_particles)
        x = x + step_var * s + np.sqrt(step_var) * z
        traj[k] = x

    return traj


def run():
    traj = reverse_sample(N_TRAJECTORIES)

    ks = np.arange(0, L + 1, 5)
    xs = np.linspace(*X_RANGE, N_X)
    heatmap = np.array([marginal_pdf(xs, sigmas[L - k]) for k in ks])

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.pcolormesh(ks, xs, heatmap.T, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="$p_i(x)$")

    colors = ["white", "silver", "dimgray", "black", "darkgray"]
    for j in range(N_TRAJECTORIES):
        ax.plot(np.arange(L + 1), traj[:, j], color=colors[j % len(colors)], linewidth=1)

    ax.set_ylim(X_RANGE[1], X_RANGE[0])
    ax.set_xlabel("reverse iteration")
    ax.set_ylabel("x")
    ax.set_title("SMLD reverse VE-SDE: single (wide) Gaussian $\\to$ Gaussian mixture")
    fig.tight_layout()

    out_path = "smld_reverse.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    run()
