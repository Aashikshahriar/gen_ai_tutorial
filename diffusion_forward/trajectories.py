"""
Replicates Figure 2.7: forward-diffusion trajectories over the heatmap of p_t(x).

Closed form (eq. 2.13), with constant alpha_t = alpha so that bar_alpha_t = alpha^t:

    x_t ~ p_t(x) = sum_k pi_k N(x | sqrt(alpha^t) mu_k, (1 - alpha^t) + alpha^t sigma_k^2)

The background heatmap plots p_t(x) as a function of (t, x). A handful of individual
sample paths are drawn on top by simulating the SDE update directly:

    x_t = sqrt(alpha) x_{t-1} + sqrt(1 - alpha) eps,   eps ~ N(0, 1)
"""

import numpy as np
import matplotlib.pyplot as plt

# --- mixture / diffusion settings (same as Example 2.1) ---
PI = (0.3, 0.7)
MU0 = (-2.0, 2.0)
SIGMA0 = (0.2, 1.0)
ALPHA = 0.97

T_MAX = 200
N_TRAJECTORIES = 6
X_RANGE = (-5, 5)
N_X = 400

rng = np.random.default_rng(0)


def normal_pdf(x, mu, var):
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)


def p_t(x, t):
    """eq. (2.13): closed-form mixture density at time t (t=0 is the initial mixture)."""
    bar_alpha = ALPHA ** t
    density = np.zeros_like(x)
    for pi_k, mu_k, sigma_k in zip(PI, MU0, SIGMA0):
        mean = np.sqrt(bar_alpha) * mu_k
        var = (1 - bar_alpha) + bar_alpha * sigma_k ** 2
        density += pi_k * normal_pdf(x, mean, var)
    return density


def sample_x0(n):
    comps = rng.choice(len(PI), size=n, p=PI)
    mus = np.array(MU0)[comps]
    sigmas = np.array(SIGMA0)[comps]
    return rng.normal(mus, sigmas)


def simulate_trajectories(n, t_max):
    x = sample_x0(n)
    traj = np.zeros((t_max + 1, n))
    traj[0] = x
    for t in range(1, t_max + 1):
        eps = rng.normal(size=n)
        x = np.sqrt(ALPHA) * x + np.sqrt(1 - ALPHA) * eps
        traj[t] = x
    return traj


def run():
    ts = np.arange(0, T_MAX + 1)
    xs = np.linspace(*X_RANGE, N_X)
    X, T = np.meshgrid(xs, ts)
    heatmap = np.array([p_t(xs, t) for t in ts])

    traj = simulate_trajectories(N_TRAJECTORIES, T_MAX)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.pcolormesh(ts, xs, heatmap.T, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="$p_t(x)$")

    for i in range(N_TRAJECTORIES):
        ax.plot(ts, traj[:, i], color="white", alpha=0.6, linewidth=1)

    ax.invert_yaxis()
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title("Realizations of random trajectories $x_t$ over $p_t(x)$")
    fig.tight_layout()

    out_path = "trajectories.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    run()
