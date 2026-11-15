import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder as Imageloader
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import gradient_penalty
from WGANmodel import Discriminator, Generator, initialize_weights


# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.00001
BATCH_SIZE = 16
IMAGE_SIZE = 32  
CHANNELS_IMG = 3
Z_DIM = 100
num_epochs = 5
FEATURES_D = 32
FEATURES_G = 32
critical_iteration = 5
lambda_gp = 10


transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = Imageloader(root="D:/cats", transform=transforms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_G).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_D).to(device)
initialize_weights(gen)
initialize_weights(critic)

optimizer_gen = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.0, 0.9))
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.9))
fixed_noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
writer_real = SummaryWriter(f"WGAN/runs/WGAN-MNIST/real")
writer_fake = SummaryWriter(f"WGAN/runs/WGAN-MNIST/fake")
step = 0

gen.train()
critic.train()

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)

        for _ in range(critical_iteration):
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, real, fake, device)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_gp * gp
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            optimizer_critic.step()


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

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img, features_d, 4, 2, 1),      # 32 -> 16
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1),    # 16 -> 8
            self._block(features_d*2, features_d*4, 4, 2, 1),  # 8 -> 4
            nn.Conv2d(features_d*4, 1, 4, 1, 0)                # 4 -> 1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super().__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, features_g*4, 4, 1, 0),         # 1x1 -> 4x4
            self._block(features_g*4, features_g*2, 4, 2, 1),  # 4x4 -> 8x8
            self._block(features_g*2, features_g, 4, 2, 1),    # 8x8 -> 16x16
            nn.ConvTranspose2d(features_g, channels_img, 4, 2, 1), # 16x16 -> 32x32
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.gen(x)