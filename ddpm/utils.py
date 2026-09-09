import os

import torch
import matplotlib.pyplot as plt


def plot_loss_curve(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history["loss"]) + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["loss"], marker="o", color="blue", label="Denoising loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DDPM Training Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.pdf"), dpi=400)
    plt.close()
    print(f"Saved loss curve to {out_dir}")


@torch.no_grad()
def plot_generated_samples(samples, out_dir, n=16, filename="generated_samples.pdf"):
    """samples: (n, 1, 28, 28) tensor, the final x0 from reverse diffusion."""
    os.makedirs(out_dir, exist_ok=True)
    samples = samples.cpu().numpy()

    grid_cols = int(n ** 0.5)
    grid_rows = (n + grid_cols - 1) // grid_cols

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(1.5 * grid_cols, 1.5 * grid_rows))
    axes = axes.flatten() if n > 1 else [axes]
    for i in range(n):
        axes[i].imshow(samples[i, 0], cmap="gray")
        axes[i].axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle("Generated samples via reverse diffusion ($x_T \\sim N(0, I) \\to x_0$)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=150)
    plt.close()
    print(f"Saved generated-sample visualization to: {os.path.join(out_dir, filename)}")


@torch.no_grad()
def plot_reverse_trajectory(snapshots, out_dir, filename="reverse_trajectory.pdf"):
    """
    snapshots: list of (t, x) pairs, x a single-image (1, 28, 28) tensor,
    showing one sample's path from xT back to x0.
    """
    os.makedirs(out_dir, exist_ok=True)
    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(1.5 * n, 1.8))
    for ax, (t, x) in zip(axes, snapshots):
        ax.imshow(x.cpu().numpy()[0], cmap="gray")
        ax.set_title(f"t={t}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Reverse diffusion trajectory ($x_T \\to x_0$)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=150)
    plt.close()
    print(f"Saved reverse-trajectory visualization to: {os.path.join(out_dir, filename)}")
