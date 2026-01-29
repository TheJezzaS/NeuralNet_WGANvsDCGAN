import torch
import torch.nn as nn


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

