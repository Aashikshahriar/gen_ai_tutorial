"""
DDPM training loss, noise-prediction (epsilon) parameterization:

    for each x0 in the batch, pick t ~ Uniform[1, T], draw
        eps ~ N(0, I),  x_t = sqrt(bar_alpha_t) x0 + sqrt(1 - bar_alpha_t) eps,
    then take a gradient step on  || eps_theta(x_t, t) - eps ||^2

This is the parameterization used in Ho et al. 2020 (the original DDPM paper);
it is mathematically equivalent to the x0-prediction loss (they differ by a
per-t reweighting), but tends to train more stably in practice.
"""


def denoising_loss(eps_hat, eps):
    return ((eps_hat - eps) ** 2).mean()
