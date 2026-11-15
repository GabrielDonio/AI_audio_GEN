import torch 
import torch.nn as nn

def gradient_penalty(critic, real, fake, device):
    BATCH_SIZE ,C,H,W= real.shape
    espilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated= espilon * real + (1 - espilon) * fake

    mixed_scores = critic(interpolated)
    gradient = torch.autograd.grad(
        inputs=interpolated,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores).to(device),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_penalty = ((gradient.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty