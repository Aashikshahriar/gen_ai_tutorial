"""
VAE model definition.

Matches the parameterization in Theorem 1.4:
    - Encoder q_phi(z|x) = N(z; mu_phi(x), sigma_phi(x)^2 * I)
      where sigma_phi(x) is a SCALAR (isotropic covariance), not per-dimension.
    - Decoder p_theta(x|z) = N(x; f_theta(z), sigma_dec^2 * I)
      where sigma_dec is a fixed (non-trained) hyperparameter.

Reparameterization: z = mu_phi(x) + sigma_phi(x) * epsilon,  epsilon ~ N(0, I_d)
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Maps x -> (mu_phi(x), log_sigma_phi(x))
    mu_phi(x)        : shape (batch, latent_dim)
    log_sigma_phi(x) : shape (batch, 1)   <-- scalar per sample, per the theorem
    """

    def __init__(self, in_channels=1, img_size=28, latent_dim=16, base_channels=32):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1),   # 28 -> 14
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),  # 14 -> 7
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),  # 7 -> 4
            nn.ReLU(inplace=True),
        )
        self.flatten_dim = base_channels * 4 * 4 * 4  # 4x4 spatial after 3 strided convs on 28x28

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        # single scalar log-sigma per sample
        self.fc_log_sigma = nn.Linear(self.flatten_dim, 1)

    def forward(self, x):
        h = self.conv(x)
        h = h.flatten(start_dim=1)
        mu = self.fc_mu(h)
        log_sigma = self.fc_log_sigma(h)  # (batch, 1), scalar per sample
        return mu, log_sigma


class Decoder(nn.Module):
    """
    Maps z -> f_theta(z), a reconstructed image in [0, 1].
    """

    def __init__(self, out_channels=1, img_size=28, latent_dim=16, base_channels=32):
        super().__init__()
        self.base_channels = base_channels
        self.init_size = 4  # matches encoder's final spatial size (4x4)

        self.fc = nn.Linear(latent_dim, base_channels * 4 * self.init_size * self.init_size)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 3, stride=2,
                                padding=1, output_padding=0),  # 4 -> 7
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 3, stride=2,
                                padding=1, output_padding=1),  # 7 -> 14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, out_channels, 3, stride=2,
                                padding=1, output_padding=1),  # 14 -> 28
            nn.Sigmoid(),  # pixels in [0, 1]
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), self.base_channels * 4, self.init_size, self.init_size)
        x_hat = self.deconv(h)
        return x_hat


class VAE(nn.Module):
    def __init__(self, in_channels=1, img_size=28, latent_dim=16, base_channels=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(in_channels, img_size, latent_dim, base_channels)
        self.decoder = Decoder(in_channels, img_size, latent_dim, base_channels)

    def reparameterize(self, mu, log_sigma, M=1):
        """
        Draw M Monte Carlo samples of z per input x.

        mu        : (batch, d)
        log_sigma : (batch, 1)
        returns z : (M, batch, d), sigma: (batch, 1)
        """
        sigma = torch.exp(log_sigma)  # (batch, 1)
        batch, d = mu.shape
        eps = torch.randn(M, batch, d, device=mu.device, dtype=mu.dtype)
        z = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps  # broadcast sigma over d
        return z, sigma

    def forward(self, x, M=1):
        mu, log_sigma = self.encoder(x)
        z, sigma = self.reparameterize(mu, log_sigma, M=M)  # z: (M, batch, d)

        M_, batch, d = z.shape
        z_flat = z.reshape(M_ * batch, d)
        x_hat_flat = self.decoder(z_flat)
        x_hat = x_hat_flat.reshape(M_, batch, *x_hat_flat.shape[1:])  # (M, batch, C, H, W)

        return x_hat, mu, sigma

    @torch.no_grad()
    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.latent_dim, device=device)
        x_hat = self.decoder(z)
        return x_hat