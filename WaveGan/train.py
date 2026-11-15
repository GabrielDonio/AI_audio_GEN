import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import gradient_penalty, save_checkpoint, load_checkpoint
from WaveGan_model import WaveGANDiscriminator, WaveGANGenerator, weights_init
import Datasetaudio as data
from Visualisation import Visualisation
def main():
    # Hyperparameters etc.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 8
    SLICE_LEN = 16384
    NB_CHANNELS = 1
    Z_DIM = 100
    MODEL_SIZE=16
    PHASE_SHUFFLE_RAD = 2
    NUM_EPOCHS = 4
    CRITIC_ITERATIONS = 5
    LAMBDA_GP = 10

    #transforms = transforms.Compose()
    files = "A:/Program Files/Documents infaudio/AI_audio/WaveGan/drums/train"
    dataset = data.AudioDataset(files, slice_len=SLICE_LEN, sample_rate=16000)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)
    # for tensorboard plotting
    vis = Visualisation(log_dir="WaveGan/runs/experiment")

    gen = WaveGANGenerator(Z_DIM,SLICE_LEN,MODEL_SIZE,NB_CHANNELS).to(device)
    critic = WaveGANDiscriminator(SLICE_LEN,MODEL_SIZE,NB_CHANNELS,PHASE_SHUFFLE_RAD).to(device)

    gen.apply(weights_init)
    critic.apply(weights_init)
    # initializate optimizer
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.9))

    # load checkpoint if it exists
    checkpoint_path = "WaveGan/Parameters/checkpoint.pth"
    if os.path.exists(checkpoint_path):   
        load_checkpoint(gen, critic, opt_gen, opt_critic, checkpoint_path)
        print("Checkpoint loaded")

    fixed_noise = torch.randn(BATCH_SIZE, Z_DIM).to(device)

    gen.train()
    critic.train()
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        vis.log_weights(gen, tag_prefix="Generator", step=epoch)
        vis.log_weights(critic, tag_prefix="Discriminator", step=epoch)
        for batch_idx, real in enumerate(loader):
            real = real.to(device)
            cur_batch_size = real.shape[0]

            # Train Critic: max E[critic(real)] - E[critic(fake)]
            # equivalent to minimizing the negative of that
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(cur_batch_size, Z_DIM).to(device)
                fake = gen(noise)
                critic_real = critic(real).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                gp = gradient_penalty(critic, real, fake, device=device)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
                )
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            gen_fake = critic(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                Loss D: {loss_critic.item():.4f}, Loss G: {loss_gen.item():.4f}, GP: {gp.item():.4f}")
            # Print losses occasionally and print to tensorboard
            if batch_idx % 100 == 0 and batch_idx > 0:
                print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                    Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}")

                with torch.no_grad():
                    fake = gen(fixed_noise.to(device))
                    vis.log_waveforms(real, fake)
                    vis.log_weights(gen, tag_prefix="Generator", step=epoch * len(loader) + batch_idx)
                    vis.log_weights(critic, tag_prefix="Discriminator", step=epoch * len(loader) + batch_idx)
    print(f"[Epoch {epoch}] Gen Output Stats - Min: {fake.min().item():.4f}, Max: {fake.max().item():.4f}, Mean: {fake.mean().item():.4f}")
    print(f"[Epoch {epoch}] Real Output Stats - Min: {real.min().item():.4f}, Max: {real.max().item():.4f}, Mean: {real.mean().item():.4f}")
    vis.close()
    save_checkpoint(gen, critic, opt_gen, opt_critic, path=checkpoint_path)
if __name__ == "__main__":
    main()