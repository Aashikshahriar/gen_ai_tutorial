"""
Train a VAE on MNIST, following Theorem 1.4's ELBO objective exactly.

Usage:
    python train.py
    python train.py --epochs 30 --batch_size 128 --latent_dim 16 --M 1 --sigma_dec 0.1

Automatically uses Apple Silicon MPS if available, else CUDA, else CPU.
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import VAE
from losses import elbo_loss
from mnist_loader import MNISTRaw
from utils import plot_loss_curves, plot_reconstructions, plot_generated_samples


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./outputs")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--M", type=int, default=1, help="Monte Carlo samples per x in the ELBO")
    p.add_argument("--sigma_dec", type=float, default=0.1, help="fixed decoder noise std")
    p.add_argument("--kl_sign", type=str, default="standard", choices=["literal", "standard"],
                    help="'literal' = exactly as printed in Theorem 1.4 (ELBO = -recon + KL); "
                         "'standard' = usual VAE convention (ELBO = -recon - KL), recommended "
                         "for stable training. See losses.py docstring.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, device, M, sigma_dec, kl_sign):
    model.train()
    total_neg_elbo, total_recon, total_kl, n_batches = 0.0, 0.0, 0.0, 0

    for x, _ in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        optimizer.zero_grad()

        x_hat, mu, sigma = model(x, M=M)
        neg_elbo, recon_term, kl_term = elbo_loss(x, x_hat, mu, sigma, sigma_dec=sigma_dec, kl_sign=kl_sign)

        neg_elbo.backward()
        optimizer.step()

        total_neg_elbo += neg_elbo.item()
        total_recon += recon_term.item()
        total_kl += kl_term.item()
        n_batches += 1

    return (total_neg_elbo / n_batches, total_recon / n_batches, total_kl / n_batches)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")
    print(f"KL sign convention: {args.kl_sign}")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    train_dataset = MNISTRaw(root=args.data_dir, train=True)
    test_dataset = MNISTRaw(root=args.data_dir, train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    model = VAE(in_channels=1, img_size=28, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"neg_elbo": [], "recon_term": [], "kl_term": []}

    for epoch in range(1, args.epochs + 1):
        neg_elbo, recon_term, kl_term = train_one_epoch(
            model, train_loader, optimizer, device, args.M, args.sigma_dec, args.kl_sign
        )
        history["neg_elbo"].append(neg_elbo)
        history["recon_term"].append(recon_term)
        history["kl_term"].append(kl_term)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"-ELBO: {neg_elbo:.4f} | recon: {recon_term:.4f} | kl: {kl_term:.4f}")

    # --- Plots ---
    plot_loss_curves(history, args.out_dir)

    # reconstructions on a batch of real test images
    x_batch, _ = next(iter(test_loader))
    plot_reconstructions(model, x_batch, device, args.out_dir, n=8)

    # pure generation from the prior
    plot_generated_samples(model, device, args.out_dir, n=16)

    # save model weights
    ckpt_path = os.path.join(args.out_dir, "vae_mnist.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved model weights to: {ckpt_path}")


if __name__ == "__main__":
    main()