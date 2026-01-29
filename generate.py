import argparse
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from models import Generator

def show_samples(G, title, device, z_dim, n_samples=16):
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    z = torch.randn(n_samples, z_dim, 1, 1, device=device)
    imgs = G(z).cpu()
    grid = make_grid(imgs, nrow=4, normalize=True)
    plt.imshow(grid.permute(1,2,0))
    plt.title(title)
    plt.axis("off")
    plt.show()


def main(args):
    # Detect device
    ## Use cuda if releven, otherwise mps (if mac), otherwise cpu (sad)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    G = Generator(z_dim=args.z_dim).to(device)
    G.load_state_dict(torch.load(args.checkpoint, map_location=device))
    G.eval()

    show_samples(G, args.title if hasattr(args, "title") else "Generated Samples", device, args.z_dim, args.n_samples)

    # z = torch.randn(args.n_samples, args.z_dim, 1, 1, device=device)
    # imgs = G(z).cpu()
    #
    # grid = make_grid(imgs, nrow=4, normalize=True)
    # plt.imshow(grid.permute(1,2,0))
    # plt.axis("off")
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--show_real", action="store_true", help="Show real images")
    parser.add_argument("--label", type=int, default=0, help="Label of real images to show")

    main(parser.parse_args())
