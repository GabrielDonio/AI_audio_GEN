import torch
import torch.nn as nn


def gradient_penalty(disc, real, fake, device="cpu"):
    BATCH_SIZE, C, L = real.shape
    alpha = torch.rand(BATCH_SIZE, 1, 1, device=device)

    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)

    # Critic scores
    mixed_scores = disc(interpolated)

    # Gradient wrt interpolated
    gradient = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradient = gradient.view(BATCH_SIZE, -1)
    gradient_norm = gradient.norm(2, dim=1)
    gp = torch.mean((gradient_norm - 1) ** 2)
    return gp

def save_checkpoint(gen, critic, opt_gen, opt_critic, path="checkpoint.pth"):
    checkpoint = {
        "gen_state_dict": gen.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "opt_gen_state_dict": opt_gen.state_dict(),
        "opt_critic_state_dict": opt_critic.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"✅ Checkpoint saved at {path}")

def load_checkpoint(gen, critic, opt_gen, opt_critic, path="checkpoint.pth"):
    checkpoint = torch.load(path)
    gen.load_state_dict(checkpoint["gen_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])
    opt_gen.load_state_dict(checkpoint["opt_gen_state_dict"])
    opt_critic.load_state_dict(checkpoint["opt_critic_state_dict"])
    print(f"✅ Checkpoint loaded at {path}")