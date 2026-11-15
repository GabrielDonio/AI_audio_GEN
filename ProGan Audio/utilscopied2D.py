import torch
import random
import numpy as np
import os
import torchaudio
import config2Dcopied as config  # adapte ce fichier aussi à l'audio si besoin
from scipy.stats import truncnorm

def plot_to_tensorboard(writer, loss_critic, loss_gen, real, fake, tensorboard_step):
    writer.add_scalar("Loss Critic", loss_critic, global_step=tensorboard_step)
    writer.add_scalar("Loss Generator", loss_gen, global_step=tensorboard_step)

    real_audio = real[:8]  # prendre jusqu’à 8 exemples
    fake_audio = fake[:8]

    for i in range(len(real_audio)):
        # Normalize to [-1, 1] before saving
        real_norm = real_audio[i] / (real_audio[i].abs().max() + 1e-8)
        fake_norm = fake_audio[i] / (fake_audio[i].abs().max() + 1e-8)
        writer.add_audio(f"Real_{i}", real_norm, global_step=tensorboard_step, sample_rate=16000)
        writer.add_audio(f"Fake_{i}", fake_norm, global_step=tensorboard_step, sample_rate=16000)


def gradient_penalty(critic, real, fake, alpha, steps, device="cpu"):
    batch_size, channels, seq_len = real.shape
    epsilon = torch.rand((batch_size, 1, 1), device=device).expand_as(real)
    interpolated = real * epsilon + fake * (1 - epsilon)
    interpolated.requires_grad_(True)

    # Passe interpolée dans le critic
    mixed_scores = critic(interpolated, alpha, steps)

    # Gradients par rapport à interpolé
    grad = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Calcul de la norme avec sécurité
    grad = grad.view(grad.size(0), -1)
    grad_norm = grad.norm(2, dim=1)

    # Ajout d'une epsilon pour éviter les NaNs
    gp = torch.mean((grad_norm - 1) ** 2 + 1e-12)
    return gp



def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_examples(gen, steps, truncation=0.7, n=100):
    gen.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            noise = torch.tensor(
                truncnorm.rvs(-truncation, truncation, size=(1, config.Z_DIM, 1)),
                device=config.DEVICE,
                dtype=torch.float32,
            )
            audio = gen(noise, alpha, steps).squeeze(0).cpu()
            # Normalize to [-1, 1] before saving
            audio = audio / (audio.abs().max() + 1e-8)
            os.makedirs("saved_examples", exist_ok=True)
            torchaudio.save(f"saved_examples/audio_{i}.wav", audio, sample_rate=16000)
    gen.train()
