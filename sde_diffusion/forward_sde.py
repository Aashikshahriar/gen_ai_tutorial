"""
Replicates Definition 4.1 (Forward Diffusion SDE) and shows it reduces to the
DDPM forward process we already built in closed form (diffusion_forward/).

    dx = f(x, t) dt + g(t) dw                                   (Eq. 4.10)

simulated by Euler-Maruyama:

    x_{k+1} = x_k + f(x_k, t_k) * dt + g(t_k) * sqrt(dt) * eps_k,   eps_k ~ N(0, I)

DDPM is the discretization of the "Variance Preserving" (VP) SDE:

    f(x, t) = -0.5 * beta(t) * x
    g(t)    = sqrt(beta(t))

with beta(t) linear in t. This SDE has a known closed-form marginal (the
continuous-time analogue of DDPM's eq. (2.13)):

    bar_alpha(t) = exp( -integral_0^t beta(s) ds )
    x(t) | x(0) ~ N( sqrt(bar_alpha(t)) x(0), (1 - bar_alpha(t)) I )

so for the same 1D Gaussian-mixture initial condition used in Example 2.1, the
marginal p_t(x) is again a two-component Gaussian mixture with unchanged
weights and t-dependent per-component mean/variance -- exactly mirroring
gmm_forward_diffusion.py's eq. (2.2) recursion, just in continuous time.

This script simulates M particles under the SDE via Euler-Maruyama and checks
their histogram at a few times against that closed form.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- mixture / SDE settings (same mixture as Example 2.1) ---
PI = (0.3, 0.7)
MU0 = (-2.0, 2.0)
SIGMA0 = (0.2, 1.0)

BETA_MIN, BETA_MAX = 0.1, 20.0   # linear beta(t) schedule, t in [0, 1]
T_MAX = 1.0
N_STEPS = 1000                   # Euler-Maruyama steps
DT = T_MAX / N_STEPS

M = 20000                        # number of simulated particles
T_SNAPSHOTS = [0.0, 0.05, 0.15, 0.3, 1.0]
X_RANGE = (-5, 5)

rng = np.random.default_rng(0)


def beta(t):
    return BETA_MIN + t * (BETA_MAX - BETA_MIN)


def bar_alpha(t):
    """exp(-integral_0^t beta(s) ds), closed form for the linear schedule."""
    integral = BETA_MIN * t + 0.5 * (BETA_MAX - BETA_MIN) * t ** 2
    return np.exp(-integral)


def f_drift(x, t):
    return -0.5 * beta(t) * x


def g_diffusion(t):
    return np.sqrt(beta(t))


def sample_x0(n):
    comps = rng.choice(len(PI), size=n, p=PI)
    mus = np.array(MU0)[comps]
    sigmas = np.array(SIGMA0)[comps]
    return rng.normal(mus, sigmas)


def simulate_sde(n_particles, n_steps, dt):
    x = sample_x0(n_particles)
    snapshots = {}
    t = 0.0
    if 0.0 in T_SNAPSHOTS:
        snapshots[0.0] = x.copy()

    for step in range(n_steps):
        eps = rng.normal(size=n_particles)
        x = x + f_drift(x, t) * dt + g_diffusion(t) * np.sqrt(dt) * eps
        t += dt
        for t_snap in T_SNAPSHOTS:
            if t_snap > 0 and abs(t - t_snap) < dt / 2:
                snapshots[t_snap] = x.copy()

    return snapshots


def closed_form_pdf(x, t):
    """Continuous-time analogue of eq. (2.13): marginal p_t(x) under the VP-SDE."""
    ba = bar_alpha(t)
    density = np.zeros_like(x)
    for pi_k, mu_k, sigma_k in zip(PI, MU0, SIGMA0):
        mean = np.sqrt(ba) * mu_k
        var = (1 - ba) + ba * sigma_k ** 2
        density += pi_k * np.exp(-0.5 * (x - mean) ** 2 / var) / np.sqrt(2 * np.pi * var)
    return density


def run():
    snapshots = simulate_sde(M, N_STEPS, DT)
    xs = np.linspace(*X_RANGE, 400)

    fig, axes = plt.subplots(1, len(T_SNAPSHOTS), figsize=(4 * len(T_SNAPSHOTS), 3.5))
    for ax, t in zip(axes, T_SNAPSHOTS):
        ax.hist(snapshots[t], bins=80, range=X_RANGE, density=True,
                color="gold", edgecolor="gold", label="Euler-Maruyama samples")
        ax.plot(xs, closed_form_pdf(xs, t), color="red", linewidth=1.5, label="closed-form $p_t(x)$")
        ax.set_title(f"t = {t}")
        ax.set_xlim(*X_RANGE)
    axes[0].legend(fontsize=7)

    fig.suptitle(r"Forward diffusion SDE ($dx = f(x,t)\,dt + g(t)\,dw$): VP-SDE $\equiv$ DDPM")
    fig.tight_layout()

    out_path = "forward_sde.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    run()
