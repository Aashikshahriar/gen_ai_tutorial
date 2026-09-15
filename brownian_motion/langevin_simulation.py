"""
Simulates the Langevin equation (Sec. 5.1) and checks that it reproduces the
physical predictions it's supposed to: Ornstein-Uhlenbeck velocity relaxation
and Einstein's linear-in-time mean squared displacement for the position.

    v_dot + gamma * v = Gamma(t),        Gamma(t) the Langevin force        (Eq. 5.4)

with:
    E[Gamma(t)] = 0
    E[Gamma(t) Gamma(t')] = q * delta(t - t')

Discretized via Euler-Maruyama (Gamma(t) dt -> sqrt(q * dt) * z, z ~ N(0, 1)):

    v_{k+1} = v_k * (1 - gamma * dt) + sqrt(q * dt) * z_k
    x_{k+1} = x_k + v_k * dt

v(t) is an Ornstein-Uhlenbeck process with known closed forms this script
checks the simulation against:

    stationary velocity variance:      Var[v] -> q / (2 * gamma)
    velocity autocorrelation:          E[v(t) v(t+tau)] = (q / (2 gamma)) * exp(-gamma |tau|)
    long-time mean squared displacement (Einstein):
                                        E[x(t)^2] ~ 2 D t,   D = q / (2 gamma^2)
"""

import numpy as np
import matplotlib.pyplot as plt

GAMMA = 1.0     # friction coefficient
Q = 2.0         # Langevin force intensity, E[Gamma(t)Gamma(t')] = Q delta(t-t')
DT = 0.001
T_TOTAL = 20.0
N_STEPS = int(T_TOTAL / DT)

N_SHOW = 4          # individual trajectories to plot
N_ENSEMBLE = 4000    # particles for MSD / autocorrelation statistics

D_THEORY = Q / (2 * GAMMA ** 2)                    # Einstein diffusion coefficient
V_VAR_THEORY = Q / (2 * GAMMA)                      # stationary velocity variance

rng = np.random.default_rng(0)


def simulate(n_particles, n_steps, dt, record_every=1):
    v = np.zeros(n_particles)
    x = np.zeros(n_particles)
    ts = np.arange(0, n_steps + 1, record_every) * dt
    v_hist = np.zeros((len(ts), n_particles))
    x_hist = np.zeros((len(ts), n_particles))
    v_hist[0] = v
    x_hist[0] = x

    rec_idx = 1
    for k in range(n_steps):
        z = rng.normal(size=n_particles)
        x = x + v * dt
        v = v * (1 - GAMMA * dt) + np.sqrt(Q * dt) * z
        if (k + 1) % record_every == 0:
            v_hist[rec_idx] = v
            x_hist[rec_idx] = x
            rec_idx += 1

    return ts, v_hist, x_hist


def run():
    record_every = 10
    ts, v_hist, x_hist = simulate(N_ENSEMBLE, N_STEPS, DT, record_every=record_every)

    msd = np.mean(x_hist ** 2, axis=1)
    v_var = np.var(v_hist, axis=1)

    # velocity autocorrelation from the (stationary, late-time) portion of the ensemble
    burn = len(ts) // 2
    v_stationary = v_hist[burn:]
    max_lag = 300
    lags = np.arange(max_lag) * DT * record_every
    autocorr = np.array([
        np.mean(v_stationary[0] * v_stationary[lag]) for lag in range(max_lag)
        if lag < v_stationary.shape[0]
    ])
    lags = lags[:len(autocorr)]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (a) individual velocity trajectories
    ax = axes[0, 0]
    show_ts, v_show, _ = simulate(N_SHOW, N_STEPS, DT, record_every=record_every)
    for j in range(N_SHOW):
        ax.plot(show_ts, v_show[:, j], linewidth=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel("v(t)")
    ax.set_title("Velocity: Ornstein-Uhlenbeck process")

    # (b) individual position trajectories (the actual "Brownian motion")
    ax = axes[0, 1]
    _, _, x_show = simulate(N_SHOW, N_STEPS, DT, record_every=record_every)
    for j in range(N_SHOW):
        ax.plot(show_ts, x_show[:, j], linewidth=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.set_title("Position: erratic pollen-grain-like motion")

    # (c) mean squared displacement vs Einstein's prediction
    ax = axes[1, 0]
    ax.plot(ts, msd, label="simulated $E[x(t)^2]$", color="tab:blue")
    ax.plot(ts, 2 * D_THEORY * ts, "--", color="black", label=f"Einstein: $2Dt$, $D$={D_THEORY:.2f}")
    ax.set_xlabel("t")
    ax.set_ylabel("mean squared displacement")
    ax.set_title("Einstein relation: MSD grows linearly in $t$")
    ax.legend(fontsize=8)

    # (d) velocity autocorrelation vs theory
    ax = axes[1, 1]
    ax.plot(lags, autocorr, label="simulated $E[v(t)v(t+\\tau)]$", color="tab:red")
    ax.plot(lags, V_VAR_THEORY * np.exp(-GAMMA * lags), "--", color="black",
            label=f"theory: $(q/2\\gamma) e^{{-\\gamma \\tau}}$")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("velocity autocorrelation")
    ax.set_title("Velocity autocorrelation decays exponentially")
    ax.legend(fontsize=8)

    fig.suptitle("Langevin equation $\\dot v + \\gamma v = \\Gamma(t)$: simulation vs. theory")
    fig.tight_layout()

    out_path = "langevin_simulation.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")
    print(f"Stationary velocity variance: simulated={v_var[-1]:.3f}, theory={V_VAR_THEORY:.3f}")
    print(f"MSD slope near t={T_TOTAL}: simulated={msd[-1]/ts[-1]:.3f}, theory (2D)={2*D_THEORY:.3f}")


if __name__ == "__main__":
    run()
