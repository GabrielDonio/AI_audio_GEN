import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from WGANmodel import Discriminator, Generator, initialize_weights
from torchvision.datasets import ImageFolder


# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.00001
BATCH_SIZE = 16
IMAGE_SIZE = 64  # MNIST original size, change to 64 if you want upscaling
CHANNELS_IMG = 3     # images couleur
Z_DIM = 100
num_epochs = 5
FEATURES_D = 64
FEATURES_G = 64
critical_iteration = 5
weight_clip = 0.01

transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*CHANNELS_IMG, [0.5]*CHANNELS_IMG)
])

# Charge dataset
dataset = ImageFolder(root="D:/cats", transform=transforms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_G).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_D).to(device)
initialize_weights(gen)
initialize_weights(critic)

optimizer_gen = torch.optim.RMSprop(gen.parameters(), lr=lr)
optimizer_critic = torch.optim.RMSprop(critic.parameters(), lr=lr)
fixed_noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
writer_real = SummaryWriter(f"WGAN/runs/WGAN-MNIST/real")
writer_fake = SummaryWriter(f"WGAN/runs/WGAN-MNIST/fake")
step = 0

gen.train()
critic.train()
print("Nombre d'images :", len(dataset))
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)

        for _ in range(critical_iteration):
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            optimizer_critic.step()
            for p in critic.parameters():
                p.data.clamp_(-weight_clip, weight_clip)

        # Générateur
        noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
        fake = gen(noise)
        output = critic(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()

        if batch_idx == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} "
                  f"Loss D: {loss_critic:.4f}, Loss G: {loss_gen:.4f}")

            with torch.no_grad():
                fake = gen(fixed_noise).detach().cpu()
                writer_fake.add_images("MNIST Fake Images", fake, global_step=step)
                real_cpu = real.cpu()
                writer_real.add_images("MNIST Real Images", real_cpu, global_step=step)
                step += 1