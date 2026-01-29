import datetime
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 256, 7, 1, 0),  # 1×1 → 7×7
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),    # 7×7 → 14×14
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 1, 4, 2, 1),      # 14×14 → 28×28
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, use_bn=True):
        super().__init__()

        layers = [
            nn.Conv2d(1, 64, 4, 2, 1),   # 28×28 → 14×14
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, 4, 2, 1) # 14×14 → 7×7
        ]

        if use_bn:
            layers.append(nn.BatchNorm2d(128))

        layers += [
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 1, 7, 1, 0)  # 7×7 → 1×1
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).view(-1)


bce = nn.BCEWithLogitsLoss()



def dcgan_d_loss(D, real, fake):
    return (
        bce(D(real), torch.ones_like(D(real))) +
        bce(D(fake), torch.zeros_like(D(fake)))
    )

def dcgan_g_loss(D, fake):
    return bce(D(fake), torch.ones_like(D(fake)))


def gradient_penalty(D, real, fake, device):
    epsilon = torch.rand(real.size(0), 1, 1, 1, device=device)
    x_hat = epsilon * real + (1 - epsilon) * fake
    x_hat.requires_grad_(True)

    D_x_hat = D(x_hat)
    grads = torch.autograd.grad(
        outputs=D_x_hat,
        inputs=x_hat,
        grad_outputs=torch.ones_like(D_x_hat),
        create_graph=True,
        retain_graph=True
    )[0]

    grads = grads.view(grads.size(0), -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


def wgan_d_loss(real_scores, fake_scores, gp, lambda_gp=10):
    return fake_scores.mean() - real_scores.mean() + lambda_gp * gp


def wgan_g_loss(fake_scores):
    return -fake_scores.mean()


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



if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.FashionMNIST(
        root="./data", train=True, transform=transform, download=True
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True)


    # Detect device
    ## Use cuda if releven, otherwise mps (if mac), otherwise cpu (sad)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # DCGAN
    G_dcgan = Generator().to(device)
    D_dcgan = Discriminator(use_bn=True).to(device)
    g_opt_dcgan = torch.optim.Adam(G_dcgan.parameters(), 2e-4, betas=(0.5, 0.999))
    d_opt_dcgan = torch.optim.Adam(D_dcgan.parameters(), 2e-4, betas=(0.5, 0.999))

    # WGAN-GP
    G_wgan = Generator().to(device)
    D_wgan = Discriminator(use_bn=False).to(device)
    g_opt_wgan = torch.optim.Adam(G_wgan.parameters(), 1e-4, betas=(0.0, 0.9))
    d_opt_wgan = torch.optim.Adam(D_wgan.parameters(), 1e-4, betas=(0.0, 0.9))


    # Train
    G_losses_dcgan, D_losses_dcgan = train_gan(
        loader, G_dcgan, D_dcgan,
        g_opt_dcgan, d_opt_dcgan,
        device, mode="dcgan"
    )

    G_losses_wgan, D_losses_wgan = train_gan(
        loader, G_wgan, D_wgan,
        g_opt_wgan, d_opt_wgan,
        device, mode="wgan"
    )


    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(G_losses_dcgan, label="G")
    plt.plot(D_losses_dcgan, label="D")
    plt.title("DCGAN Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(G_losses_wgan, label="G")
    plt.plot(D_losses_wgan, label="D")
    plt.title("WGAN-GP Loss")
    plt.legend()

    plt.show()






    def show_samples(G, title):
        z = torch.randn(16, 128, 1, 1, device=device)
        imgs = G(z).cpu()
        grid = make_grid(imgs, nrow=4, normalize=True)
        plt.imshow(grid.permute(1,2,0))
        plt.title(title)
        plt.axis("off")
        plt.show()


    show_samples(G_dcgan, "DCGAN Samples")
    show_samples(G_wgan, "WGAN-GP Samples")


    label = 0  # T-shirt
    real_imgs = [img for img, y in dataset if y == label][:2]
    grid = make_grid(real_imgs, nrow=2, normalize=True)
    plt.imshow(grid.permute(1,2,0))
    plt.title("Real Images (Label 0)")
    plt.axis("off")
    plt.show()



