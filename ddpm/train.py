"""
Train a DDPM denoiser on MNIST, following the training algorithm exactly:

    for each x0:
        repeat until convergence:
            t ~ Uniform[1, T]
            x_t = sqrt(bar_alpha_t) x0 + sqrt(1 - bar_alpha_t) eps,  eps ~ N(0, I)
            take a gradient step on || x_theta(x_t, t) - x0 ||^2

After training, samples are drawn by reverse diffusion starting from
x_T ~ N(0, I) and recursively applying the DDPM posterior update down to x_0.

Usage:
    python train.py
    python train.py --epochs 20 --batch_size 128 --T 300
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Denoiser
from diffusion import DiffusionSchedule
from losses import denoising_loss
from mnist_loader import MNISTRaw
from utils import plot_loss_curve, plot_generated_samples, plot_reverse_trajectory


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
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--T", type=int, default=300, help="number of diffusion timesteps")
    p.add_argument("--beta_1", type=float, default=1e-4)
    p.add_argument("--beta_T", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def train_one_epoch(model, schedule, loader, optimizer, device):
    model.train()
    total_loss, n_batches = 0.0, 0

    for x0, _ in tqdm(loader, desc="train", leave=False):
        x0 = x0.to(device)
        batch = x0.shape[0]
        t = torch.randint(1, schedule.T + 1, (batch,), device=device)

        x_t, _ = schedule.q_sample(x0, t)
        x0_hat = model(x_t, t)
        loss = denoising_loss(x0_hat, x0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def sample(model, schedule, num_samples, device, keep_trajectory_for=0):
    model.eval()
    x_t = torch.randn(num_samples, 1, 28, 28, device=device)
    snapshots = []
    if keep_trajectory_for:
        snapshots.append((schedule.T, x_t[keep_trajectory_for - 1].clone()))

    for t_int in range(schedule.T, 0, -1):
        x_t, x0_hat = schedule.reverse_step(model, x_t, t_int)
        if keep_trajectory_for and t_int % max(1, schedule.T // 10) == 0:
            snapshots.append((t_int - 1, x_t[keep_trajectory_for - 1].clone()))

    return x_t, snapshots


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    train_dataset = MNISTRaw(root=args.data_dir, train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False)

    schedule = DiffusionSchedule(T=args.T, beta_1=args.beta_1, beta_T=args.beta_T, device=device)
    model = Denoiser(in_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"loss": []}
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, schedule, train_loader, optimizer, device)
        history["loss"].append(loss)
        print(f"Epoch {epoch:3d}/{args.epochs} | denoising MSE: {loss:.4f}")

    plot_loss_curve(history, args.out_dir)

    samples, snapshots = sample(model, schedule, num_samples=16, device=device, keep_trajectory_for=1)
    plot_generated_samples(samples, args.out_dir)
    plot_reverse_trajectory(snapshots, args.out_dir)

    ckpt_path = os.path.join(args.out_dir, "ddpm_mnist.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved model weights to: {ckpt_path}")


if __name__ == "__main__":
    main()
