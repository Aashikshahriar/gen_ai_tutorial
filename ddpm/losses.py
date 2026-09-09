"""
DDPM training loss, simplified via stochastic gradient descent as described:

    for each x0 in the batch, pick t ~ Uniform[1, T], draw x_t ~ N(x_t | sqrt(bar_alpha_t) x0, (1 - bar_alpha_t) I),
    then take a gradient step on  || x_theta(x_t) - x0 ||^2
"""


def denoising_loss(x0_hat, x0):
    return ((x0_hat - x0) ** 2).mean()
