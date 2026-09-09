"""
Replicates Remark 1: comparing SGD, (full-batch) Langevin Dynamics, and
Stochastic Gradient Langevin Dynamics (SGLD) on a Bayesian MAP / posterior
estimation problem.

Toy model: estimate a scalar mean theta from N observations
    theta ~ N(0, tau^2)                      (prior)
    y_i | theta ~ N(theta, sigma^2), i=1..N  (likelihood)

so the posterior p(theta | y) is available in closed form (conjugate Gaussian)
and we can compare each method's trajectory/samples against ground truth.

Update rules (n is the minibatch size, N the full dataset size):

    SGD:              Delta theta_t = (eps_t / 2) * ( grad log p(theta_t)
                                        + (N/n) sum_{i in batch} grad log p(y_i | theta_t) )

    Langevin Dynamics: Delta theta_t = eps * ( grad log p(theta_t)
                                        + sum_{i=1}^N grad log p(y_i | theta_t) ) + eta_t

    SG Langevin Dyn.: Delta theta_t = (eps_t / 2) * ( grad log p(theta_t)
                                        + (N/n) sum_{i in batch} grad log p(y_i | theta_t) ) + eta_t

    eta_t ~ N(0, eps_t)  (injected Langevin noise)

SGD has no injected noise and, with minibatches, converges to (fluctuates around)
the MAP estimate — a single point, not a distribution. Langevin Dynamics uses the
full-data gradient plus noise and asymptotically samples the true posterior. SGLD
substitutes the cheaper minibatch gradient into the same noisy update, trading a
bit of accuracy for the ability to scale to large N.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- toy Bayesian model ---
THETA_TRUE = 2.0
SIGMA = 1.0        # observation noise std
TAU = 5.0          # prior std (fairly uninformative)
N = 1000           # full dataset size
BATCH_N = 20       # minibatch size for SGD / SGLD

T_ITERS = 3000
BURN_IN = 500       # discard the first BURN_IN samples when forming Langevin/SGLD histograms
THETA_0 = -8.0       # shared starting point for all three methods

rng = np.random.default_rng(0)


def make_data():
    return rng.normal(THETA_TRUE, SIGMA, size=N)


def grad_log_prior(theta):
    return -theta / TAU ** 2


def grad_log_likelihood(theta, y):
    """sum_i grad_theta log p(y_i | theta) over the given data y (full batch or minibatch)."""
    return np.sum(y - theta) / SIGMA ** 2


def true_posterior():
    precision = 1.0 / TAU ** 2 + N / SIGMA ** 2
    mean = (np.sum(DATA) / SIGMA ** 2) / precision
    var = 1.0 / precision
    return mean, var


def step_size_schedule(t, eps0=0.0015, decay=0.999):
    """A simple decaying step size, common in SGLD to satisfy Robbins-Monro conditions."""
    return eps0 * (decay ** t)


def run_sgd():
    theta = THETA_0
    traj = np.zeros(T_ITERS)
    for t in range(T_ITERS):
        eps_t = step_size_schedule(t)
        batch = rng.choice(DATA, size=BATCH_N, replace=False)
        grad = grad_log_prior(theta) + (N / BATCH_N) * grad_log_likelihood(theta, batch)
        theta = theta + (eps_t / 2) * grad
        traj[t] = theta
    return traj


def run_langevin(eps=2e-4):
    """Full-batch Langevin dynamics: uses ALL N data points every step, plus noise."""
    theta = THETA_0
    traj = np.zeros(T_ITERS)
    for t in range(T_ITERS):
        grad = grad_log_prior(theta) + grad_log_likelihood(theta, DATA)
        noise = rng.normal(0, np.sqrt(eps))
        theta = theta + eps * grad + noise
        traj[t] = theta
    return traj


def run_sgld():
    theta = THETA_0
    traj = np.zeros(T_ITERS)
    for t in range(T_ITERS):
        eps_t = step_size_schedule(t)
        batch = rng.choice(DATA, size=BATCH_N, replace=False)
        grad = grad_log_prior(theta) + (N / BATCH_N) * grad_log_likelihood(theta, batch)
        noise = rng.normal(0, np.sqrt(eps_t))
        theta = theta + (eps_t / 2) * grad + noise
        traj[t] = theta
    return traj


def run():
    global DATA
    DATA = make_data()
    post_mean, post_var = true_posterior()
    post_std = np.sqrt(post_var)

    sgd_traj = run_sgd()
    langevin_traj = run_langevin()
    sgld_traj = run_sgld()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    iters = np.arange(T_ITERS)
    ax = axes[0]
    ax.plot(iters, sgd_traj, color="tab:blue", label="SGD", linewidth=1)
    ax.plot(iters, langevin_traj, color="tab:red", label="Langevin Dynamics", linewidth=1, alpha=0.8)
    ax.plot(iters, sgld_traj, color="tab:green", label="SG Langevin Dynamics", linewidth=1, alpha=0.8)
    ax.axhline(post_mean, color="black", linestyle="--", linewidth=1, label="True posterior mean")
    ax.set_xlabel("iteration $t$")
    ax.set_ylabel(r"$\theta_t$")
    ax.set_title("Trajectories")
    ax.legend(fontsize=8)

    ax = axes[1]
    theta_grid = np.linspace(post_mean - 5 * post_std, post_mean + 5 * post_std, 400)
    true_pdf = np.exp(-0.5 * ((theta_grid - post_mean) / post_std) ** 2) / (post_std * np.sqrt(2 * np.pi))
    ax.plot(theta_grid, true_pdf, color="black", linewidth=1.5, label="True posterior $p(\\theta \\mid y)$")
    ax.hist(langevin_traj[BURN_IN:], bins=60, density=True, color="tab:red", alpha=0.5,
            label="Langevin Dynamics samples")
    ax.hist(sgld_traj[BURN_IN:], bins=60, density=True, color="tab:green", alpha=0.5,
            label="SGLD samples")
    ax.axvline(sgd_traj[-1], color="tab:blue", linewidth=2, label="SGD final estimate (point, no uncertainty)")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("density")
    ax.set_title("Posterior approximation (post burn-in)")
    ax.legend(fontsize=8)

    fig.suptitle("SGD vs. Langevin Dynamics vs. SG Langevin Dynamics")
    fig.tight_layout()

    out_path = "sgld_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")
    print(f"True posterior: mean={post_mean:.3f}, std={post_std:.3f}")
    print(f"SGD final theta: {sgd_traj[-1]:.3f}")
    print(f"Langevin post-burn-in mean/std: {langevin_traj[BURN_IN:].mean():.3f} / {langevin_traj[BURN_IN:].std():.3f}")
    print(f"SGLD post-burn-in mean/std: {sgld_traj[BURN_IN:].mean():.3f} / {sgld_traj[BURN_IN:].std():.3f}")


if __name__ == "__main__":
    run()
