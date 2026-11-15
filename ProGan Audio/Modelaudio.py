import torch
import torch.nn as nn
import torch.nn.functional as F

factors = [1, 2, 4, 8, 16, 16, 16]

class WSConv1D(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=7, stride=1, padding=3, gain=2
    ):
        super(WSConv1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * kernel_size)) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1)

class PixelNorm(nn.Module): #works with 1D data
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        # Pixelwise normalization: divide by sqrt(mean squared value)
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super(ConvBlock, self).__init__()
        self.use_pn = use_pixelnorm
        self.conv1 = WSConv1D(in_channels, out_channels)
        self.conv2 = WSConv1D(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pn else x
        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pn else x
        return x

class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, audio_channels=1):
        super(Generator, self).__init__()

        # initial takes 1x1 -> 1024x1 (sequence length 1024)
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose1d(z_dim, in_channels, kernel_size=256, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            WSConv1D(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )

        self.initial_rgb = WSConv1D(
            in_channels, audio_channels, kernel_size=1, stride=1, padding=0
        )
        self.prog_blocks, self.rgb_layers = (
            nn.ModuleList([]),
            nn.ModuleList([self.initial_rgb]),
        )

        for i in range(len(factors) - 1):  # same factors logic
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i + 1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
            self.rgb_layers.append(
                WSConv1D(conv_out_c, audio_channels, kernel_size=1, stride=1, padding=0)
            )

    def fade_in(self, alpha, upscaled, generated):
        out = (1 - alpha) * upscaled + alpha * generated
        return torch.tanh(out)

    def forward(self, x, alpha, steps):
        out = self.initial(x)  # out shape: (B, in_channels, 1024)

        if steps == 0:
            return self.initial_rgb(out)

        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode="linear", align_corners=False)
            out = self.prog_blocks[step](upscaled)

        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)
        return self.fade_in(alpha, final_upscaled, final_out)


class Discriminator(nn.Module):
    def __init__(self, z_dim, in_channels, audio_channels=1):
        super(Discriminator, self).__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        # On crée les blocs en partant de la résolution la plus grande (ici 1024),
        # en miroir du générateur.
        for i in range(len(factors) - 1, 0, -1):
            conv_in = int(in_channels * factors[i])
            conv_out = int(in_channels * factors[i - 1])
            self.prog_blocks.append(ConvBlock(conv_in, conv_out, use_pixelnorm=False))
            self.rgb_layers.append(
                WSConv1D(audio_channels, conv_in, kernel_size=1, stride=1, padding=0)
            )

        # couche pour la plus petite résolution (1024 samples ici)
        self.initial_rgb = WSConv1D(
            audio_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)

        # Bloc final adapté à une entrée de taille 1024 (et non 4)
        self.final_block = nn.Sequential(
            WSConv1D(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            # kernel_size = 1024 pour réduire dimension temporelle 1024 -> 1
            WSConv1D(in_channels, in_channels, kernel_size=256, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WSConv1D(in_channels, 1, kernel_size=1, padding=0, stride=1),
        )

    def fade_in(self, alpha, downscaled, out):
        """Fade in entre l'entrée downsamplée et la sortie du CNN"""
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        # x: (B, C, T)
        batch_std = torch.std(x, dim=0, keepdim=True)  # (1, C, T)
        mean_std = batch_std.mean(dim=1, keepdim=True)  # (1, 1, T)
        mean_std = mean_std.repeat(x.shape[0], 1, 1)  # (B, 1, T)
        return torch.cat([x, mean_std], dim=1)  # concat sur la dimension des canaux

    def forward(self, x, alpha, steps):
        cur_step = len(self.prog_blocks) - steps

        out = self.leaky(self.rgb_layers[cur_step](x))

        if steps == 0:  # input est de taille 1024 ici (pas 4)
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)
        
        downscaled = self.avg_pool(self.leaky(self.rgb_layers[cur_step + 1](x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))

        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)

if __name__ == "__main__":
    # === Paramètres de test ===
    batch_size = 8
    z_dim = 512
    in_channels = 16
    audio_channels = 1
    steps = 6  # étape de progression (résolution)
    alpha = 0.5

    # === Instanciation des modèles ===
    G = Generator(z_dim=z_dim, in_channels=in_channels, audio_channels=audio_channels)
    D = Discriminator(z_dim=z_dim, in_channels=in_channels, audio_channels=audio_channels)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n=== PARAMÈTRES DU MODÈLE ===")
    print(f"Nombre de paramètres du Generator : {count_parameters(G):,}")
    print(f"Nombre de paramètres du Discriminator : {count_parameters(D):,}")
    # === Génération d'un batch latent ===
    z = torch.randn(batch_size, z_dim, 1)  # (B, z_dim, 1)

    print("\n=== TEST DU GENERATOR ===")
    fake_audio = G(z, alpha=alpha, steps=steps)  # (B, 1, T)
    print("Fake audio shape:", fake_audio.shape)
    print("Fake audio min/max:", fake_audio.min().item(), fake_audio.max().item())
    print("Contient NaNs ?", torch.isnan(fake_audio).any().item())

    print("\n=== TEST DU DISCRIMINATOR ===")
    output = D(fake_audio, alpha=alpha, steps=steps)  # (B, 1)
    print("Discriminator output shape:", output.shape)
    print("Discriminator output:", output)
    print("Contient NaNs ?", torch.isnan(output).any().item())

    # === Optionnel : torchinfo (pour voir les shapes des modules) ===
    # summary(G, input_size=(batch_size, z_dim, 1), col_names=["input_size", "output_size", "num_params"])
    # summary(D, input_size=(batch_size, 1, fake_audio.shape[-1]), col_names=["input_size", "output_size", "num_params"])
