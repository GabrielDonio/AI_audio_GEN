""" Training of ProGAN using WGAN-GP loss adapted for 1D audio """
import os
from tqdm import tqdm
import torch
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utilscopied2D import (  # adapte ce fichier pour audio
    gradient_penalty,
    plot_to_tensorboard,
    save_checkpoint,
    load_checkpoint,
    generate_examples,
)
from Modelaudio import Discriminator, Generator  # à adapter en Model1D
from math import log2
import config2Dcopied as config  # à adapter en config1D
import audioDataset
import torchaudio.transforms as T

torch.backends.cudnn.benchmark = True

def get_loader(step, config):
    base_seq_len = 16384  # taille max dans le dataset (1 seconde à 16kHz)
    
    # calcul de la taille progressive
    seq_len = 256 * 2 ** step  # 256, 512, 1024, ...
    batch_size = config.BATCH_SIZES[step]

    dataset = audioDataset.AudioDataset(root=config.DATASET, max_len=base_seq_len)  # pas de trunc/padding ici

    # Calcul de la nouvelle fréquence d'échantillonnage pour resampler vers seq_len
    new_freq = int(16000 * seq_len / base_seq_len)
    downsampler = T.Resample(orig_freq=16000, new_freq=new_freq)

    def collate_fn(batch):
        waveforms = torch.stack([item[0] for item in batch])  # (B, C, base_seq_len)
        waveforms = downsampler(waveforms)  # resample vers la nouvelle taille seq_len approx.
        # Pas de trim ni padding, on fait confiance au resampler
        return waveforms, torch.zeros(len(batch))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=config.NUM_WORKERS, collate_fn=collate_fn)

    return loader, dataset, seq_len

def train_fn(
    critic,
    gen,
    loader,
    dataset,
    step,
    alpha,
    opt_critic,
    opt_gen,
    tensorboard_step,
    writer,
    scaler_gen,
    scaler_critic,
):
    loop = tqdm(loader)
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(config.DEVICE)
        cur_batch_size = real.shape[0]

        # bruit de forme (batch, z_dim, 1)
        noise = torch.randn(cur_batch_size, config.Z_DIM, 1).to(config.DEVICE)

        with torch.amp.autocast(device_type=config.DEVICE):
            fake = gen(noise, alpha, step)
            critic_real = critic(real, alpha, step)
            critic_fake = critic(fake.detach(), alpha, step)
            gp = gradient_penalty(critic, real, fake, alpha, step, device=config.DEVICE)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + config.LAMBDA_GP * gp
                + (0.001 * torch.mean(critic_real ** 2))
            )

        # Debug: check for NaN
        if (
            torch.isnan(loss_critic)
            or torch.isnan(gp)
            or torch.isnan(critic_real).any()
            or torch.isnan(critic_fake).any()
        ):
            print('NaN detected in loss_critic or gp!')
            print('loss_critic:', loss_critic)
            print('gp:', gp)
            print('critic_real:', critic_real)
            print('critic_fake:', critic_fake)
            print('real:', real)
            print('fake:', fake)
            break

        opt_critic.zero_grad()
        scaler_critic.scale(loss_critic).backward()
        scaler_critic.step(opt_critic)
        scaler_critic.update()

        with torch.amp.autocast(device_type=config.DEVICE):
            # Re-generate fake for generator update to avoid using the same graph
            fake_for_gen = gen(noise, alpha, step)
            gen_fake = critic(fake_for_gen, alpha, step)
            loss_gen = -torch.mean(gen_fake)

        if torch.isnan(loss_gen) or torch.isnan(gen_fake).any():
            print('NaN detected in loss_gen!')
            print('loss_gen:', loss_gen)
            print('gen_fake:', gen_fake)
            break

        opt_gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        alpha += cur_batch_size / ((config.PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset))
        alpha = min(alpha, 1)

        if batch_idx % 500 == 0:
            with torch.no_grad():
                fixed_fakes = gen(config.FIXED_NOISE, alpha, step) * 0.5 + 0.5
            plot_to_tensorboard(
                writer,
                loss_critic.item(),
                loss_gen.item(),
                real.detach(),
                fixed_fakes.detach(),
                tensorboard_step,
            )
            tensorboard_step += 1

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
        )

    return tensorboard_step, alpha


def main():
    gen = Generator(
        config.Z_DIM, config.IN_CHANNELS, audio_channels=config.CHANNELS_AUDIO
    ).to(config.DEVICE)
    critic = Discriminator(
        config.Z_DIM, config.IN_CHANNELS, audio_channels=config.CHANNELS_AUDIO
    ).to(config.DEVICE)

    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    scaler_critic = torch.amp.GradScaler()
    scaler_gen = torch.amp.GradScaler()

    writer = SummaryWriter(f"ProGan audio/logs")

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC, critic, opt_critic, config.LEARNING_RATE)

    gen.train()
    critic.train()

    tensorboard_step = 0
    step = int(log2(config.START_TRAIN_AT_SEQ_LEN / 256))  # ex : 16384 seq_len
    for num_epochs in config.PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5
        seq_len = 256 * 2 ** step
        loader, dataset, seq_len = get_loader(step, config)
        freq = 16000 * seq_len / 16384  # calcul de la fréquence d'échantillonnage réelle si besoin

        print(f"Current sequence length: {seq_len}")
        print("train_fn lancé") 
        for epoch in range(num_epochs):
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            tensorboard_step, alpha = train_fn(
                critic,
                gen,
                loader,
                dataset,
                step,
                alpha,
                opt_critic,
                opt_gen,
                tensorboard_step,
                writer,
                scaler_gen,
                scaler_critic,
            )

            if config.SAVE_MODEL:
                save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
                save_checkpoint(critic, opt_critic, filename=config.CHECKPOINT_CRITIC)

        step += 1


if __name__ == "__main__":
    main()
