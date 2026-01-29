import argparse
import torch
import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models import Generator, Discriminator
from losses import *


def train_gan(
    loader, G, D, g_opt, d_opt, device,
    mode="wgan", z_dim=128,
    epochs=20, n_critic=5
):

    G_losses, D_losses = [], []
    for epoch in range(epochs):
        epoch_start_time = datetime.datetime.now()
        for i, (real_img, _) in enumerate(loader):  # real= the image, its called real cause later we care about it as apose to synthetic images
            real_img = real_img.to(device) # the _ is the label, which we dont care about here

            # ------------------ Discriminator ------------------
            for _ in range(n_critic if mode == "wgan" else 1):

                z = torch.randn(real_img.size(0), z_dim, 1, 1, device=device)
                fake_img = G(z).detach() # detatch removes the gradients from the object, now its just an image, not connected to G
                                         # . VERY IMPORTANT FOR BACKPROP

                if mode == "dcgan":
                    d_loss = dcgan_d_loss(D, real_img, fake_img)
                else:
                    # d_loss = Dw(x˜) − Dw(x) + λ(||∇xˆ Dw(xˆ)||_2 − 1)^2
                    gp = gradient_penalty(D, real_img, fake_img, device)
                    real_scores = D(real_img)
                    fake_scores = D(fake_img)
                    d_loss = wgan_d_loss(real_scores, fake_scores, gp)

                d_opt.zero_grad()
                d_loss.backward()
                d_opt.step()

            # ------------------ Generator ------------------
            z = torch.randn(real_img.size(0), z_dim, 1, 1, device=device)
            fake_img = G(z)

            g_loss = (
                dcgan_g_loss(D, fake_img)
                if mode == "dcgan"
                else wgan_g_loss(D(fake_img))
            )

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())

        print(f"{mode.upper()} Epoch {epoch} \ntook {datetime.datetime.now()-epoch_start_time}")


    return G_losses, D_losses

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )

    # Models
    G = Generator(z_dim=args.z_dim).to(device)
    D = Discriminator().to(device)

    # Optimizers
    if args.mode == "dcgan":
        g_opt = torch.optim.Adam(G.parameters(), args.lr, betas=(0.5, 0.999))
        d_opt = torch.optim.Adam(D.parameters(), args.lr, betas=(0.5, 0.999))
    else:
        g_opt = torch.optim.Adam(G.parameters(), args.lr, betas=(0.5, 0.9))
        d_opt = torch.optim.Adam(D.parameters(), args.lr, betas=(0.5, 0.9))

    # Train
    G_losses, D_losses = train_gan(
        loader=loader,
        G=G,
        D=D,
        g_opt=g_opt,
        d_opt=d_opt,
        device=device,
        mode=args.mode,
        z_dim=args.z_dim,
        epochs=args.epochs,
        n_critic=args.n_critic
    )

    # Save checkpoints
    torch.save(G.state_dict(), f"checkpoints/G_{args.mode}.pth")
    torch.save(D.state_dict(), f"checkpoints/D_{args.mode}.pth")

    print("Training finished. Models saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["dcgan", "wgan"], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--n_critic", type=int, default=5)

    args = parser.parse_args()
    main(args)
