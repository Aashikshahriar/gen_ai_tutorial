"""
Replicates Theorem 4.3: the SMLD (score-matching Langevin dynamics /
NCSN-style) forward process written as a Variance-Exploding (VE) SDE.

Discrete SMLD forward: x_i = x_0 + sigma_i * z,   z ~ N(0, I), with a
geometric noise schedule sigma_i (sigma_0 ~ 0, growing to sigma_max).

Taking the continuous-time limit as in the proof of Theorem 4.3 gives:

    dx = sqrt( d[sigma(t)^2] / dt ) dw                            (Eq. 4.19)

Unlike the VP-SDE (DDPM), there is NO drift term (f(x,t) = 0) -- the data is
never rescaled, only progressively buried in additive noise of growing scale.
This means the marginal p_t(x) simply adds sigma(t)^2 to each mixture
component's variance, leaving the means and mixture weights untouched:

    p_t(x) = sum_k pi_k N(x | mu_k, sigma_k0^2 + sigma(t)^2)

This script simulates the VE-SDE via Euler-Maruyama and checks the resulting
histogram against that closed form, the VE-SDE analogue of what
forward_sde.py does for the VP-SDE.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- mixture / VE-SDE settings (same mixture as Example 2.1) ---
PI = (0.3, 0.7)
MU0 = (-2.0, 2.0)
SIGMA0 = (0.2, 1.0)

SIGMA_MIN, SIGMA_MAX = 0.01, 10.0  # geometric noise schedule, t in [0, 1]
T_MAX = 1.0
N_STEPS = 1000
DT = T_MAX / N_STEPS

M = 20000
T_SNAPSHOTS = [0.0, 0.2, 0.4, 0.6, 1.0]
X_RANGE = (-15, 15)

rng = np.random.default_rng(0)


def sigma_of_t(t):
    """Geometric schedule sigma(t) = sigma_min * (sigma_max / sigma_min)^t."""
    return SIGMA_MIN * (SIGMA_MAX / SIGMA_MIN) ** t


def g_diffusion(t):
    """g(t) = sqrt(d[sigma(t)^2]/dt) for the geometric schedule, in closed form."""
    log_ratio = np.log(SIGMA_MAX / SIGMA_MIN)
    return sigma_of_t(t) * np.sqrt(2 * log_ratio)


def sample_x0(n):
    comps = rng.choice(len(PI), size=n, p=PI)
    mus = np.array(MU0)[comps]
    sigmas = np.array(SIGMA0)[comps]
    return rng.normal(mus, sigmas)


def simulate_ve_sde(n_particles, n_steps, dt):
    x = sample_x0(n_particles)
    snapshots = {}
    t = 0.0
    if 0.0 in T_SNAPSHOTS:
        snapshots[0.0] = x.copy()

    for step in range(n_steps):
        eps = rng.normal(size=n_particles)
        x = x + g_diffusion(t) * np.sqrt(dt) * eps  # f(x,t) = 0, no drift
        t += dt
        for t_snap in T_SNAPSHOTS:
            if t_snap > 0 and abs(t - t_snap) < dt / 2:
                snapshots[t_snap] = x.copy()

    return snapshots


def closed_form_pdf(x, t):
    """Marginal p_t(x) under the VE-SDE: variance-only broadening of each component."""
    sigma_t2 = sigma_of_t(t) ** 2
    density = np.zeros_like(x)
    for pi_k, mu_k, sigma_k in zip(PI, MU0, SIGMA0):
        var = sigma_k ** 2 + sigma_t2
        density += pi_k * np.exp(-0.5 * (x - mu_k) ** 2 / var) / np.sqrt(2 * np.pi * var)
    return density


def run():
    snapshots = simulate_ve_sde(M, N_STEPS, DT)
    xs = np.linspace(*X_RANGE, 400)

    fig, axes = plt.subplots(1, len(T_SNAPSHOTS), figsize=(4 * len(T_SNAPSHOTS), 3.5))
    for ax, t in zip(axes, T_SNAPSHOTS):
        ax.hist(snapshots[t], bins=80, range=X_RANGE, density=True,
                color="gold", edgecolor="gold", label="Euler-Maruyama samples")
        ax.plot(xs, closed_form_pdf(xs, t), color="red", linewidth=1.5, label="closed-form $p_t(x)$")
        ax.set_title(f"t = {t}  ($\\sigma(t)$={sigma_of_t(t):.2f})")
        ax.set_xlim(*X_RANGE)
    axes[0].legend(fontsize=7)

    fig.suptitle(r"SMLD forward VE-SDE ($dx = g(t)\,dw$, no drift): variance keeps exploding")
    fig.tight_layout()

    out_path = "smld_forward.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    run()
