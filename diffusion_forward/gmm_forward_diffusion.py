"""
Replicates Example 2.1 / Figure 2.5: forward diffusion of a 1D Gaussian mixture.

    x_0 ~ p_0(x) = pi_1 N(x | mu_1, sigma_1^2) + pi_2 N(x | mu_2, sigma_2^2)
    x_t = sqrt(alpha_t) x_{t-1} + sqrt(1 - alpha_t) eps,   eps ~ N(0, I)

Eq. (2.2) gives the closed-form recursion for each component's mean/variance
at step t, so p_t(x) stays a two-component Gaussian mixture with fixed
weights pi_1, pi_2 for every t:

    mu_{k,t}    = sqrt(alpha_t) * mu_{k,t-1}
    sigma_{k,t}^2 = alpha_t * sigma_{k,t-1}^2 + (1 - alpha_t)

As t -> infinity, both components' means shrink to 0 and variances converge
to 1, so p_t(x) converges to a standard white Gaussian N(0, 1).
"""

import numpy as np
import matplotlib.pyplot as plt

# --- mixture / diffusion settings (matching the example) ---
PI = (0.3, 0.7)
MU0 = (-2.0, 2.0)
SIGMA0 = (0.2, 1.0)
ALPHA_T = 0.97  # constant alpha_t for all t

T_LIST = [1, 5, 10, 30, 40, 50, 100, 200]
X = np.linspace(-5, 5, 500)


def normal_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def mixture_pdf(x, mus, sigmas, pis):
    return sum(p * normal_pdf(x, mu, sigma) for p, mu, sigma in zip(pis, mus, sigmas))


def step_params(mus, vars_, alpha):
    """One application of eq. (2.2): advance (mean, variance) per component by one step."""
    new_mus = tuple(np.sqrt(alpha) * mu for mu in mus)
    new_vars = tuple(alpha * v + (1 - alpha) for v in vars_)
    return new_mus, new_vars


def run():
    mus, vars_ = MU0, tuple(s ** 2 for s in SIGMA0)
    t = 0
    snapshots = {}

    max_t = max(T_LIST)
    while t < max_t:
        mus, vars_ = step_params(mus, vars_, ALPHA_T)
        t += 1
        if t in T_LIST:
            sigmas = tuple(np.sqrt(v) for v in vars_)
            snapshots[t] = mixture_pdf(X, mus, sigmas, PI)

    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    for ax, t in zip(axes.flat, T_LIST):
        ax.plot(X, snapshots[t])
        ax.set_title(f"t = {t}")
        ax.set_xlim(-5, 5)

    fig.suptitle("Evolution of $p_t(x)$: bimodal mixture converging to a Gaussian")
    fig.tight_layout()

    out_path = "forward_diffusion.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    run()
