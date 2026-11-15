import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveGANGenerator(nn.Module):
    def __init__(self, z_dim, slice_len=16384, model_size=64, nb_channels=1):
        super().__init__()
        assert slice_len in (16384, 32768, 65536), "slice_len must be 16384, 32768 or 65536"

        # Enregistre les paramètres
        self.model_size = model_size
        self.dim_mul = 16 if slice_len == 16384 else 32

        # Projection: z -> (batch, C, 16)
        self.fc = nn.Linear(z_dim, 16 * model_size * self.dim_mul)

        # Construit les couches upsampling
        layers = []
        muls = [self.dim_mul, self.dim_mul // 2, self.dim_mul // 4, self.dim_mul // 8]

        for mul in muls:
            layers.append(self._block(
                in_ch=model_size * mul,
                out_ch=model_size * (mul // 2),
                stride=4
            ))

        if self.dim_mul == 32:
            # Ajoute un bloc supplémentaire
            layers.append(self._block(
                in_ch=model_size * (muls[-1] // 2),
                out_ch=model_size * (muls[-1] // 4),
                stride=4
            ))

        # Final stage
        layers.append(self._block(
            in_ch=model_size * (self.dim_mul // 16),
            out_ch=nb_channels,
            stride=4,
            relu=False  # Pas de ReLU à la fin pour la sortie finale
        ))

        layers.append(nn.Tanh())
        self.gen = nn.Sequential(*layers)

    def _block(self, in_ch, out_ch, stride, relu=True):
        k = 25
        output_padding = 1
        padding = self.compute_padding(k, stride, output_padding)

        layers = [
            nn.ConvTranspose1d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=k,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            )
    ]

        if relu:
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)


    def compute_padding(self, k, stride, output_padding):
        return (k - stride + output_padding) // 2

    def forward(self, z):
        batch = z.size(0)
        x = self.fc(z)                             # (B, C*16)
        x = x.view(batch, self.model_size * self.dim_mul, 16)  # (B, C, 16)
        x = torch.relu(x)
        return self.gen(x)
        
class WaveGANDiscriminator(nn.Module):
    def __init__(self, slice_len=16384, model_size=64, nb_channels=1,phase_shuffle_rad=0):
        super().__init__()
        assert slice_len in (16384, 32768, 65536), "slice_len must be 16384, 32768 or 65536"

        self.slice_len = slice_len
        self.model_size = model_size
        self.dim_mul = 16 if slice_len == 16384 else 32
        self.phase_shuffle_rad = phase_shuffle_rad

        # On construit les couches conv1d pour downsampling avec stride 4
        muls = [self.dim_mul, self.dim_mul * 2, self.dim_mul * 4, self.dim_mul * 8]
        in_channels = nb_channels
        layers = []
        for mul in muls:
            layers.append(self._block(
                in_ch=in_channels,
                out_ch=model_size * mul,
                stride=4
            ))
            in_channels = model_size * mul

        self.conv_layers = nn.Sequential(*layers)
        # 4 convolutions -> taille finale = slice_len / (4**4) = slice_len / 256
        final_length = slice_len // (4 ** len(muls))
        self.fc = nn.Linear(in_channels * final_length, 1)  # sortie scalaire par batch

    def _block(self, in_ch, out_ch, stride):
        k=25
        padding = self.compute_padding(k, stride)
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=stride, padding=padding),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def compute_padding(self,kernel_size, stride):
        return (kernel_size - stride) // 2 + (1 if (kernel_size - stride) % 2 else 0)
    
    def phase_shuffle(self, x, rad, mode='reflect'):
        if rad == 0:
            return x

        batch_size, channels, seq_len = x.shape
        phase_shifts = torch.randint(-rad, rad + 1, (batch_size,), device=x.device)

        x_shuffled = torch.zeros_like(x)

        for i, phase in enumerate(phase_shifts):
            if phase == 0:
                x_shuffled[i] = x[i]
            else:
                pad_l = max(phase, 0)
                pad_r = max(-phase, 0)
                x_padded = F.pad(x[i:i+1], (pad_l, pad_r), mode=mode)
                start = pad_r
                end = start + seq_len
                x_shuffled[i] = x_padded[:, :, start:end]

        return x_shuffled


    def forward(self, x):
        out = x
    # on applique les blocs 1 par 1 pour insérer la phase shuffle
        for i, layer in enumerate(self.conv_layers):
            out = layer(out)
            if i < len(self.conv_layers) - 1:
                out = self.phase_shuffle(out, rad=self.phase_shuffle_rad)

        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out # retourne (batch,)

def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

if __name__ == "__main__":
    # Test rapide
    batch_size = 4
    slice_len = 16384
    nb_channels = 1

    x = torch.randn(batch_size, nb_channels, slice_len)
    disc = WaveGANDiscriminator(slice_len=slice_len, model_size=64, nb_channels=nb_channels)
    out = disc(x)
    print(out.shape)  # doit être torch.Size([4])
    
if __name__ == "__main__":   
    gen = WaveGANGenerator(z_dim=100, slice_len=16384, model_size=64, nb_channels=1)
    z = torch.randn(4, 100)
    out = gen(z)
    print(out.shape)  # doit être (4, 1, 16384)
    assert out.shape == (4, 1, 16384)
if __name__ == "__main__":
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    gen = WaveGANGenerator(z_dim=100, slice_len=16384, model_size=64, nb_channels=1)
    disc = WaveGANDiscriminator(slice_len=16384, model_size=64, nb_channels=1)

    print(f"Nombre de paramètres du générateur : {count_parameters(gen):,}")
    print(f"Nombre de paramètres du discriminateur : {count_parameters(disc):,}")
