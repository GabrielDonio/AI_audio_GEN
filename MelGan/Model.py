import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
import torch.nn.functional as F

class ResidualStack(nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=(1, 3, 5)):
        super().__init__()
        self.layers = nn.ModuleList()
        for d in dilations:
            padding = (kernel_size - 1) // 2 * d
            block = nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=d)),
                nn.LeakyReLU(0.2),
                weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2)),
            )
            self.layers.append(block)

    def forward(self, x):
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual
        return x

class Generator(nn.Module):
    def __init__(self, n_mels=80, upsample_scales=(8, 8, 2, 2), initial_channels=128):
        super().__init__()
        self.initial_conv = weight_norm(nn.Conv1d(n_mels, initial_channels, kernel_size=7, padding=3))
        
        in_ch = initial_channels
        self.upsample_layers = nn.ModuleList()
        self.residual_stacks = nn.ModuleList()
        
        for scale in upsample_scales:
            out_ch = in_ch // 2
            self.upsample_layers.append(
                weight_norm(nn.ConvTranspose1d(in_ch, out_ch, kernel_size=scale*2, stride=scale, padding=scale//2))
            )
            self.residual_stacks.append(ResidualStack(out_ch))
            in_ch = out_ch
        
        self.final_conv = weight_norm(nn.Conv1d(in_ch, 1, kernel_size=7, padding=3))

    def forward(self, x):
        x = self.initial_conv(x)
        for upsample, res_stack in zip(self.upsample_layers, self.residual_stacks):
            x = F.leaky_relu(upsample(x), 0.2)
            x = res_stack(x)
        x = self.final_conv(F.leaky_relu(x, 0.2))
        return torch.tanh(x)

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.initial_conv = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels, 16, kernel_size=15, stride=1, padding=7)),
            nn.LeakyReLU(0.2)
        )
        
        self.downsample = nn.ModuleList([
            nn.Sequential(
                weight_norm(nn.Conv1d(16, 64, kernel_size=41, stride=4, padding=20, groups=4)),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(64, 256, kernel_size=41, stride=4, padding=20, groups=16)),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(256, 512, kernel_size=41, stride=4, padding=20, groups=64)),
                nn.LeakyReLU(0.2)
            ),
        ])
        
        self.final_conv1 = nn.Sequential(
            weight_norm(nn.Conv1d(512, 512, kernel_size=5, stride=1, padding=2)),
            nn.LeakyReLU(0.2)
        )
        
        self.final_conv2 = weight_norm(nn.Conv1d(512, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x = self.initial_conv(x)
        for layer in self.downsample:
            x = layer(x)
        fmap = self.final_conv1(x)
        out = self.final_conv2(fmap)
        return out, fmap

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorBlock() for _ in range(3)])

    def forward(self, x):
        outputs = []
        feature_maps = []

        x1 = x
        x2 = F.avg_pool1d(x, kernel_size=4, stride=2, padding=1)
        x3 = F.avg_pool1d(x2, kernel_size=4, stride=2, padding=1)

        for disc, inp in zip(self.discriminators, [x1, x2, x3]):
            out, fmap = disc(inp)
            outputs.append(out)
            feature_maps.append(fmap)

        return outputs, feature_maps


if __name__ == "__main__":
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    mel = torch.randn(2, 80, 10)
    gen = Generator()
    wav = gen(mel)
    print(f"Output wav shape: {wav.shape}")
    print(f"Generator params: {count_parameters(gen)/1e6:.2f}M")

    batch_size = 2
    audio_length = 16384
    x = torch.randn(batch_size, 1, audio_length)

    msd = MultiScaleDiscriminator()
    outputs, feature_maps = msd(x)
    for i, (out, fmap) in enumerate(zip(outputs, feature_maps), 1):
        print(f"Discriminator {i}: output shape = {out.shape}, fmap shape = {fmap.shape}")
    print(f"MultiScaleDiscriminator params: {count_parameters(msd)/1e6:.2f}M")
