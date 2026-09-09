import os
import torch
import matplotlib.pyplot as plt

def plot_loss_curves(history,out_dir):
    os.makedirs(out_dir, exist_ok=True)
    epochs= range(1, len(history['neg_elbo']) + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["neg_elbo"],marker="o",color="blue",label="Total Loss (-ELBO)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss (Negative ELBO)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "combined_loss.pdf"),dpi=400)
    plt.close()

    fig,axes=plt.subplots(1,2,figsize=(12,5))
    axes[0].plot(epochs, history["recon_term"],marker="o",color="red",label="Reconstruction Loss")
    axes[0].set_title("Reconstruction Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("||x - x_hat||^2 / (2*sigma_dec^2)")
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, history["kl_term"],marker="o",color="green",label="KL Divergence")
    axes[1].set_title("KL Divergence")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("KL(q(z|x)||p(z))")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_components.pdf"),dpi=400)
    plt.close()

    # all three in one plot

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["neg_elbo"],marker="o",color="blue",label="Total Loss (-ELBO)")
    plt.plot(epochs, history["recon_term"],marker="o",color="red",label="Reconstruction Loss")
    plt.plot(epochs, history["kl_term"],marker="o",color="green",label="KL Divergence")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("All Loss Components")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "all_loss_components.pdf"),dpi=400)
    plt.close()

    print(f"Saved loss plots to {out_dir}")


@torch.no_grad()


def plot_reconstructions(model,x,device, out_dir,n=8,filename="reconstructions.pdf"):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    x = x[:n].to(device)
    mu,log_sigma= model.encoder(x)
    x_hat = model.decoder(mu)
    x_cpu=x.cpu().numpy()
    x_hat_cpu=x_hat.cpu().numpy()
    fig, axes = plt.subplots(2, n, figsize=(1.5 * n, 3.2))
    for i in range(n):
        axes[0, i].imshow(x_cpu[i, 0], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(x_hat_cpu[i, 0], cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Reconstruction", fontsize=10)
    fig.suptitle("Reconstructions (top: original, bottom: reconstructed)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=150)
    plt.close()
    print(f"Saved reconstruction visualization to: {os.path.join(out_dir, filename)}")


@torch.no_grad()
def plot_generated_samples(model, device, out_dir, n=16, filename="generated_samples.pdf"):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    samples = model.sample(n, device).cpu().numpy()
 
    grid_cols = int(n ** 0.5)
    grid_rows = (n + grid_cols - 1) // grid_cols
 
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(1.5 * grid_cols, 1.5 * grid_rows))
    axes = axes.flatten() if n > 1 else [axes]
    for i in range(n):
        axes[i].imshow(samples[i, 0], cmap="gray")
        axes[i].axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")
 
    fig.suptitle("Generated samples from prior N(0, I) (unconditional generation)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=150)
    plt.close()
    print(f"Saved generated-sample visualization to: {os.path.join(out_dir, filename)}")