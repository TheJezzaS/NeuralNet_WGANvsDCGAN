import torch.nn as nn

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