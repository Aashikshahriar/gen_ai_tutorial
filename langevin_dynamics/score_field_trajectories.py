"""
Replicates Figure 3.6: the score-function vector field of a 2D Gaussian
mixture, and the deterministic trajectories of samples climbing that field.

    p(x) = pi_1 N(x | mu_1, Sigma_1) + pi_2 N(x | mu_2, Sigma_2),   x in R^2

(a) contour map of log p(x) with the score vector field grad_x log p(x)
    overlaid as arrows (score points "uphill", toward higher density).
(b) same vector field, with a couple of sample trajectories x_t obtained by
    repeatedly following the score (gradient ascent, no noise):

        x_{t+1} = x_t + eta * grad_x log p(x_t)

    Each trajectory climbs the field and settles near a mode of p(x).
"""

import numpy as np
import matplotlib.pyplot as plt

# --- 2D Gaussian mixture settings ---
PI = (0.5, 0.5)
MU = (np.array([4.0, 4.0]), np.array([-4.0, -4.0]))
SIGMA = 2.5  # isotropic std for both components

ETA = 0.5
T_STEPS = 60
START_POINTS = (np.array([-8.0, 8.0]), np.array([-8.0, -3.0]))

GRID_RANGE = (-10, 10)
GRID_N = 200
QUIVER_STRIDE = 10


def component_pdf(x, mu):
    """x: (..., 2), mu: (2,) -> (...,) """
    diff = x - mu
    sq_dist = np.sum(diff ** 2, axis=-1)
    return np.exp(-0.5 * sq_dist / SIGMA ** 2) / (2 * np.pi * SIGMA ** 2)


def log_density(x):
    p = sum(pi_k * component_pdf(x, mu_k) for pi_k, mu_k in zip(PI, MU))
    return np.log(p + 1e-12)


def score(x):
    """grad_x log p(x), x: (..., 2) -> (..., 2)"""
    comp_pdfs = [pi_k * component_pdf(x, mu_k) for pi_k, mu_k in zip(PI, MU)]
    total = sum(comp_pdfs) + 1e-12
    comp_scores = [-(x - mu_k) / SIGMA ** 2 for mu_k in MU]  # (..., 2) each
    weighted = sum(w[..., None] * cs for w, cs in zip(comp_pdfs, comp_scores))
    return weighted / total[..., None]


def run_trajectory(x0, steps=T_STEPS, eta=ETA):
    traj = np.zeros((steps + 1, 2))
    traj[0] = x0
    x = x0.copy()
    for t in range(steps):
        x = x + eta * score(x)
        traj[t + 1] = x
    return traj


def make_grid(n):
    xs = np.linspace(*GRID_RANGE, n)
    ys = np.linspace(*GRID_RANGE, n)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X, Y], axis=-1)
    return X, Y, pts


def draw_score_quiver(ax, X, Y, pts, stride=QUIVER_STRIDE, color="black"):
    S = score(pts)
    norm = np.linalg.norm(S, axis=-1, keepdims=True) + 1e-8
    S_unit = S / norm
    ax.quiver(X[::stride, ::stride], Y[::stride, ::stride],
               S_unit[::stride, ::stride, 0], S_unit[::stride, ::stride, 1],
               color=color, pivot="mid", scale=30, width=0.003)


def run():
    X, Y, pts = make_grid(GRID_N)
    logp = log_density(pts)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # (a) contour map of log p(x) + score field
    ax = axes[0]
    ax.contour(X, Y, logp, levels=25, cmap="rainbow", linewidths=1)
    draw_score_quiver(ax, X, Y, pts)
    ax.set_xlim(*GRID_RANGE)
    ax.set_ylim(*GRID_RANGE)
    ax.set_title(r"(a) vector field of $\nabla_x \log p(x)$")

    # (b) score field + sample trajectories
    ax = axes[1]
    draw_score_quiver(ax, X, Y, pts)
    for x0 in START_POINTS:
        traj = run_trajectory(x0)
        ax.plot(traj[:, 0], traj[:, 1], "r.-", markersize=6, linewidth=1)
    ax.set_xlim(*GRID_RANGE)
    ax.set_ylim(*GRID_RANGE)
    ax.set_title(r"(b) $x_t$ trajectory")

    fig.suptitle("Score function field and sample trajectories")
    fig.tight_layout()

    out_path = "score_field_trajectories.pdf"
    fig.savefig(out_path, dpi=400)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    run()
