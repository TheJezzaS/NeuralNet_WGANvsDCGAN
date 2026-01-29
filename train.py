import argparse
import torch
import datetime
import os
import sys
import numpy as np  # Added for std deviation calculation
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models import Generator, Discriminator
from losses import *

# ---------------------------------------------------------
#  Helper: Convergence & Health Check (With Early Stopping)
# ---------------------------------------------------------
def check_training_status(G_losses, D_losses, current_epoch, max_epochs, patience=10, min_delta=0.05):
    """
    Returns: (status_string, message)
    Status: FAILED, CONVERGED, TRAINING
    """
    # 1. CHECK FAILURE: NaN (Explosion)
    if len(G_losses) > 0:
        if torch.isnan(torch.tensor(G_losses[-1])) or torch.isnan(torch.tensor(D_losses[-1])):
            return "FAILED", "Loss became NaN (Exploded)"

    # 2. CHECK FAILURE: Vanishing Gradient
    if len(D_losses) > 0 and abs(D_losses[-1]) < 1e-5:
        return "FAILED", "Discriminator won completely (Loss ~ 0.0)"

    # 3. CHECK FAILURE: Frozen Generator (Mode Collapse)
    # If G loss hasn't moved at all in 50 steps (very strict)
    if len(G_losses) > 50:
        recent = G_losses[-50:]
        if max(recent) - min(recent) < 1e-6:
            return "FAILED", "Generator loss is completely frozen"

    # 4. CHECK EARLY STOPPING (Convergence)
    # We only check this if we have enough history (patience)
    if current_epoch > patience:
        # Get the recent history of losses (last 'patience' epochs)
        # Note: G_losses is a list of ALL batches. We need to average by epoch approx.
        # Simplification: We look at the average loss of the LAST few batches per epoch.
        # Better: Let's assume G_losses passed here is the average loss PER EPOCH (we will fix main loop to do this)
        
        recent_G = G_losses[-patience + 5:] # look at the last 5 epochs for statistics
        recent_D = D_losses[-patience + 5:] # look at the last 5 epochs for statistics

        # Calculate stability (Standard Deviation or Range)
        g_std = np.std(recent_G)
        d_std = np.std(recent_D)
        
        # If both losses have stabilized (variance is low), we assume convergence.
        # Note: WGAN relies mostly on D loss stability.
        if d_std < min_delta and g_std < min_delta:
            return "CONVERGED", f"Loss stabilized (std_dev < {min_delta}) over last {patience} epochs."

    # 5. CHECK MAX EPOCHS
    if current_epoch >= max_epochs:
        return "CONVERGED", "Reached maximum epochs"

    return "TRAINING", "Healthy"

# ---------------------------------------------------------
#  Main Execution
# ---------------------------------------------------------
def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    os.makedirs("checkpoints", exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    successful_run = False
    fail_count = 0
    
    # Store history for plotting
    final_G_history, final_D_history = [], []

    while not successful_run:
        print(f"\n--- Attempt {fail_count + 1} ---")
        
        G = Generator(z_dim=args.z_dim).to(device)
        D = Discriminator().to(device) 

        if args.mode == "dcgan":
            g_opt = torch.optim.Adam(G.parameters(), args.lr, betas=(0.5, 0.999))
            d_opt = torch.optim.Adam(D.parameters(), args.lr, betas=(0.5, 0.999))
            print("\nTraining DCGAN")
        else:
            g_opt = torch.optim.Adam(G.parameters(), args.lr, betas=(0.0, 0.9))
            d_opt = torch.optim.Adam(D.parameters(), args.lr, betas=(0.0, 0.9))
            print("\nTraining WGAN")
        theta_converged = False
        epoch = 0
        
        # We store EPOCH AVERAGES for the convergence check (smoother than batch losses)
        epoch_G_losses = []
        epoch_D_losses = []
        
        # We also store raw batch losses for the final plot
        batch_G_losses = []
        batch_D_losses = []

        while not theta_converged:
            epoch_start = datetime.datetime.now()
            
            # Temp lists for this epoch
            curr_epoch_g = []
            curr_epoch_d = []
            for i, (real_img, _) in enumerate(loader):
                real_img = real_img.to(device)

                # Update Critic
                for _ in range(args.n_critic if args.mode == "wgan" else 1):
                    z = torch.randn(real_img.size(0), args.z_dim, 1, 1, device=device)
                    fake_img = G(z).detach()

                    if args.mode == "dcgan":
                        d_loss = dcgan_d_loss(D, real_img, fake_img)
                    else:
                        gp = gradient_penalty(D, real_img, fake_img, device)
                        d_loss = wgan_d_loss(D(real_img), D(fake_img), gp)

                    d_opt.zero_grad()
                    d_loss.backward()
                    d_opt.step()

                # Update Generator
                z = torch.randn(real_img.size(0), args.z_dim, 1, 1, device=device)
                fake_img = G(z)
                
                g_loss = (dcgan_g_loss(D, fake_img) if args.mode == "dcgan" 
                          else wgan_g_loss(D(fake_img)))

                g_opt.zero_grad()
                g_loss.backward()
                g_opt.step()

                # Record Batch Data
                curr_epoch_g.append(g_loss.item())
                curr_epoch_d.append(d_loss.item())
                batch_G_losses.append(g_loss.item())
                batch_D_losses.append(d_loss.item())

            # --- Calculate Average for this Epoch ---
            avg_g = sum(curr_epoch_g) / len(curr_epoch_g)
            avg_d = sum(curr_epoch_d) / len(curr_epoch_d)
            
            epoch_G_losses.append(avg_g)
            epoch_D_losses.append(avg_d)
            epoch += 1

            # --- CHECK STATUS ---
            # We pass the EPOCH averages to the check function for stability analysis
            status, reason = check_training_status(
                epoch_G_losses, epoch_D_losses, 
                current_epoch=epoch, 
                max_epochs=args.epochs,
                patience=10,    # Wait at least 5 epochs
                min_delta=0.02 # Strictness of stability
            )

            if status == "FAILED":
                print(f"--> FAILURE at Epoch {epoch}: {reason}")
                fail_count += 1
                break 
            
            elif status == "CONVERGED":
                print(f"--> CONVERGED at Epoch {epoch}: {reason}")
                theta_converged = True
                successful_run = True
                final_G_history, final_D_history = batch_G_losses, batch_D_losses
            
            else:
                print(f"{args.mode.upper()} Epoch {epoch}/{args.epochs} | Avg Loss D: {avg_d:.4f} G: {avg_g:.4f} | run time: {datetime.datetime.now() - epoch_start}")

    print(f"\nDone. Total failures: {fail_count}")
    
    torch.save(G.state_dict(), f"checkpoints/G_{args.mode}.pth")
    torch.save(D.state_dict(), f"checkpoints/D_{args.mode}.pth")

    plt.figure(figsize=(10,4))
    plt.plot(final_G_history, label="G Loss", alpha=0.7)
    plt.plot(final_D_history, label="D Loss", alpha=0.7)
    plt.title(f"{args.mode.upper()} Loss (Stopped at Epoch {len(epoch_G_losses)})")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dcgan", "wgan"], required=True)
    parser.add_argument("--epochs", type=int, default=20) # Set this high!
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--n_critic", type=int, default=5)

    args = parser.parse_args()
    main(args)