"""
Numerically solves the Fokker-Planck equation for the velocity process defined
by the Langevin equation from brownian_motion/langevin_simulation.py:

    v_dot + gamma * v = Gamma(t),   E[Gamma(t)Gamma(t')] = q * delta(t - t')

The Fokker-Planck equation (the PDE satisfied by p(v, t), the probability
density of v_t, as opposed to sampling individual SDE trajectories) is:

    dp(v,t)/dt = -d/dv[ f(v) p(v,t) ] + (1/2) q d^2 p(v,t)/dv^2,   f(v) = -gamma * v

i.e. drift term (advection, from f) plus diffusion term (from the noise
intensity q). This is solved directly on a grid via an explicit finite
difference scheme (central differences in v, forward Euler in t) -- no
particles, no sampling, just marching the PDE forward in time.

Starting from a near-delta initial condition p(v, 0) = N(v; 0, sigma_0^2)
(sigma_0 small), the known closed-form solution is the OU transition density:

    p(v, t) = N( v ; 0, (q / (2 gamma)) (1 - exp(-2 gamma t)) + sigma_0^2 exp(-2 gamma t) )

which relaxes to the stationary distribution N(0, q / (2 gamma)) as t -> inf.
This script checks the PDE solution against that closed form, and against
the Monte Carlo histogram from directly simulating the Langevin SDE -- three
independent descriptions of the same process (SDE sample paths, the PDE they
solve, and the analytic solution) that should all agree.
"""

import numpy as np
import matplotlib.pyplot as plt

GAMMA = 1.0
Q = 2.0
SIGMA0 = 0.05  # width of the near-delta initial condition

V_RANGE = (-6, 6)
NV = 400

T_SNAPSHOTS = [0.0, 0.25, 0.5, 1.0, 3.0]
T_MAX = max(T_SNAPSHOTS)

rng = np.random.default_rng(0)


def analytic_pdf(v, t):
    var = (Q / (2 * GAMMA)) * (1 - np.exp(-2 * GAMMA * t)) + SIGMA0 ** 2 * np.exp(-2 * GAMMA * t)
    return np.exp(-0.5 * v ** 2 / var) / np.sqrt(2 * np.pi * var)


def solve_fokker_planck():
    v = np.linspace(*V_RANGE, NV)
    dv = v[1] - v[0]

    # explicit-scheme stability: dt must satisfy both the diffusion and advection CFL conditions
    dt = 0.2 * min(dv ** 2 / Q, dv / (GAMMA * abs(V_RANGE[0])))
    n_steps = int(np.ceil(T_MAX / dt))
    dt = T_MAX / n_steps

    p = np.exp(-0.5 * v ** 2 / SIGMA0 ** 2) / np.sqrt(2 * np.pi * SIGMA0 ** 2)
    p /= np.sum(p) * dv  # normalize

    f = -GAMMA * v  # drift f(v)

    snapshots = {}
    t = 0.0
    if 0.0 in T_SNAPSHOTS:
        snapshots[0.0] = p.copy()

    for step in range(n_steps):
        flux = f * p
        dflux_dv = np.zeros_like(p)
        dflux_dv[1:-1] = (flux[2:] - flux[:-2]) / (2 * dv)

        d2p_dv2 = np.zeros_like(p)
        d2p_dv2[1:-1] = (p[2:] - 2 * p[1:-1] + p[:-2]) / dv ** 2

        p = p + dt * (-dflux_dv + 0.5 * Q * d2p_dv2)
        p[0] = p[-1] = 0.0  # absorbing boundary (domain wide enough that p ~ 0 there anyway)

        t += dt
        for t_snap in T_SNAPSHOTS:
            if t_snap > 0 and abs(t - t_snap) < dt / 2:
                snapshots[t_snap] = p.copy()

    return v, snapshots


def simulate_sde(n_particles, t_snapshots, dt=0.001):
    v = rng.normal(0, SIGMA0, size=n_particles)
    n_steps = int(np.ceil(max(t_snapshots) / dt))
    dt = max(t_snapshots) / n_steps

    snapshots = {}
    t = 0.0
    if 0.0 in t_snapshots:
        snapshots[0.0] = v.copy()

    for step in range(n_steps):
        z = rng.normal(size=n_particles)
        v = v * (1 - GAMMA * dt) + np.sqrt(Q * dt) * z
        t += dt
        for t_snap in t_snapshots:
            if t_snap > 0 and abs(t - t_snap) < dt / 2:
                snapshots[t_snap] = v.copy()

    return snapshots


def run():
    v, fp_snapshots = solve_fokker_planck()
    sde_snapshots = simulate_sde(20000, T_SNAPSHOTS)

    fig, axes = plt.subplots(1, len(T_SNAPSHOTS), figsize=(4 * len(T_SNAPSHOTS), 3.5))
    for ax, t in zip(axes, T_SNAPSHOTS):
        ax.hist(sde_snapshots[t], bins=60, range=V_RANGE, density=True,
                color="gold", edgecolor="gold", label="SDE Monte Carlo", alpha=0.8)
        ax.plot(v, fp_snapshots[t], color="blue", linewidth=2, label="Fokker-Planck PDE")
        ax.plot(v, analytic_pdf(v, t), "--", color="red", linewidth=1.5, label="analytic $p(v,t)$")
        ax.set_title(f"t = {t}")
        ax.set_xlim(*V_RANGE)
    axes[0].legend(fontsize=7)

    fig.suptitle("Fokker-Planck equation for the Langevin/OU velocity process: three ways, one answer")
    fig.tight_layout()

    out_path = "fokker_planck_ou.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")

    for t in T_SNAPSHOTS:
        fp_var = np.trapezoid(fp_snapshots[t] * v ** 2, v)
        print(f"t={t:.2f} | Fokker-Planck var={fp_var:.4f} | "
              f"analytic var={(Q/(2*GAMMA))*(1-np.exp(-2*GAMMA*t)) + SIGMA0**2*np.exp(-2*GAMMA*t):.4f} | "
              f"SDE var={np.var(sde_snapshots[t]):.4f}")


if __name__ == "__main__":
    run()
