import torch


def elbo_loss(x, x_hat, mu, sigma, sigma_dec=0.1, kl_sign="literal"):
    """
    x       : (batch, C, H, W)      ground-truth input
    x_hat   : (M, batch, C, H, W)   M reconstructions per input (Monte Carlo samples)
    mu      : (batch, d)            encoder mean
    sigma   : (batch, 1)            encoder scalar std (already exponentiated)
    sigma_dec : float               fixed decoder noise std
    kl_sign : "literal"  -> ELBO = -recon_term + kl_term   (exactly as printed in the theorem)
              "standard" -> ELBO = -recon_term - kl_term   (the usual VAE convention, since
                             kl_term as computed here IS KL(q_phi(z|x) || N(0,I)), and a
                             standard ELBO subtracts the KL divergence, it doesn't add it)

    Returns:
        neg_elbo   : scalar tensor to minimize (this IS the training loss)
        recon_term : scalar, mean reconstruction cost term
        kl_term    : scalar, mean KL(q||p) term (always returned as a positive divergence,
                     regardless of kl_sign, so it plots sensibly)
    """
    M, batch = x_hat.shape[0], x_hat.shape[1]
    d = mu.shape[1]

    # --- Reconstruction term: (1/M) sum_m ||x - x_hat_m||^2 / (2 sigma_dec^2) ---
    x_expand = x.unsqueeze(0).expand(M, *x.shape)               # (M, batch, C, H, W)
    sq_err = (x_expand - x_hat).pow(2).flatten(start_dim=2).sum(dim=2)  # (M, batch)
    recon_per_sample = sq_err.mean(dim=0) / (2 * sigma_dec ** 2)        # (batch,)

    # --- KL term: (1/2)( sigma^2 * d - d + ||mu||^2 - 2*d*log(sigma) ) ---
    sigma = sigma.squeeze(-1)                                    # (batch,)
    mu_sq_norm = mu.pow(2).sum(dim=1)                            # (batch,)
    kl_per_sample = 0.5 * (sigma.pow(2) * d - d + mu_sq_norm - 2 * d * torch.log(sigma + 1e-8))

    if kl_sign == "literal":
        elbo_per_sample = -recon_per_sample + kl_per_sample
    elif kl_sign == "standard":
        elbo_per_sample = -recon_per_sample - kl_per_sample
    else:
        raise ValueError(f"Unknown kl_sign: {kl_sign!r}, expected 'literal' or 'standard'")

    neg_elbo = -elbo_per_sample.mean()   # what the optimizer minimizes

    recon_term = recon_per_sample.mean()
    kl_term = kl_per_sample.mean()

    return neg_elbo, recon_term, kl_term