"""
Forward noising schedule and the DDPM training / reverse-sampling math.

Forward process (variable alpha_t per step):
    beta_t linear schedule in [beta_1, beta_T]
    alpha_t     = 1 - beta_t
    bar_alpha_t = prod_{i=1}^{t} alpha_i

Forward sample (used to build training pairs (x0, xt)):
    x_t = sqrt(bar_alpha_t) x0 + sqrt(1 - bar_alpha_t) eps,   eps ~ N(0, I)

Reverse step (DDPM posterior q(x_{t-1} | x_t, x0), x0 replaced by the denoiser's
prediction x0_hat = x_theta(x_t, t) at inference time):
    mu_q(x_t, x0)  = [ sqrt(alpha_t)(1 - bar_alpha_{t-1}) / (1 - bar_alpha_t) ] x_t
                   + [ sqrt(bar_alpha_{t-1})(1 - alpha_t) / (1 - bar_alpha_t) ] x0
    sigma_q(t)^2   = (1 - alpha_t)(1 - bar_alpha_{t-1}) / (1 - bar_alpha_t)

    x_{t-1} = mu_q(x_t, x0_hat) + sigma_q(t) * eps,   eps ~ N(0, I)  (eps = 0 at t = 1)
"""

import torch


class DiffusionSchedule:
    def __init__(self, T=300, beta_1=1e-4, beta_T=0.02, device="cpu"):
        self.T = T
        betas = torch.linspace(beta_1, beta_T, T, device=device)  # index 0 -> t=1
        alphas = 1.0 - betas
        bar_alphas = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.bar_alphas = bar_alphas
        # bar_alpha_{t-1}, with bar_alpha_0 := 1
        self.bar_alphas_prev = torch.cat([torch.ones(1, device=device), bar_alphas[:-1]])

    def _gather(self, arr, t):
        """t is a (batch,) LongTensor of 1-indexed timesteps; arr is 0-indexed by (t-1)."""
        out = arr[t - 1]
        return out.view(-1, 1, 1, 1)

    def q_sample(self, x0, t, eps=None):
        """Draw x_t ~ q(x_t | x0) for a batch of (x0, t) pairs."""
        if eps is None:
            eps = torch.randn_like(x0)
        bar_alpha_t = self._gather(self.bar_alphas, t)
        x_t = torch.sqrt(bar_alpha_t) * x0 + torch.sqrt(1 - bar_alpha_t) * eps
        return x_t, eps

    def posterior_mean_var(self, x_t, x0_hat, t):
        """mu_q(x_t, x0_hat) and sigma_q(t)^2 for a batch of (x_t, t), t all equal here."""
        alpha_t = self._gather(self.alphas, t)
        bar_alpha_t = self._gather(self.bar_alphas, t)
        bar_alpha_prev = self._gather(self.bar_alphas_prev, t)

        coef_xt = torch.sqrt(alpha_t) * (1 - bar_alpha_prev) / (1 - bar_alpha_t)
        coef_x0 = torch.sqrt(bar_alpha_prev) * (1 - alpha_t) / (1 - bar_alpha_t)
        mean = coef_xt * x_t + coef_x0 * x0_hat

        var = (1 - alpha_t) * (1 - bar_alpha_prev) / (1 - bar_alpha_t)
        return mean, var

    @torch.no_grad()
    def reverse_step(self, model, x_t, t_int):
        """One reverse step t -> t-1 (t_int is a python int, same for the whole batch)."""
        batch = x_t.shape[0]
        t = torch.full((batch,), t_int, dtype=torch.long, device=x_t.device)

        x0_hat = model(x_t, t)
        mean, var = self.posterior_mean_var(x_t, x0_hat, t)

        if t_int > 1:
            eps = torch.randn_like(x_t)
        else:
            eps = torch.zeros_like(x_t)
        x_prev = mean + torch.sqrt(var) * eps
        return x_prev, x0_hat
