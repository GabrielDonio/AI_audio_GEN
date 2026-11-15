import torch
import torch.utils.data as Data
import librosa

class WaveNetDataset(Data.Dataset):
    def __init__(self, file_path, slice_len=16384, sample_rate=16000, mu=255):
        super(WaveNetDataset, self).__init__()
        self.slice_len = slice_len
        self.sample_rate = sample_rate
        self.mu = mu

        # Load audio
        try:
            waveform, sr = librosa.load(file_path, sr=sample_rate, mono=True)
            self.waveform = torch.tensor(waveform, dtype=torch.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to load {file_path}: {e}")

        self.encoded = self.mu_law_encode(self.waveform)

    def mu_law_encode(self, x):
        mu = self.mu
        x = torch.clamp(x, -1.0, 1.0)
        fx = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(torch.tensor(mu, dtype=torch.float32))
        encoded = ((fx + 1) / 2 * mu + 0.5).long()
        return encoded

    @staticmethod
    def mu_law_decode(x, mu=255):
        x = x.float()
        fx = 2 * (x / mu) - 1
        decoded = torch.sign(fx) * (1 / mu) * ((1 + mu) ** torch.abs(fx) - 1)
        return decoded

    def __len__(self):
        return self.encoded.size(0) - self.slice_len

    def __getitem__(self, idx):
        x = self.encoded[idx : idx + self.slice_len].unsqueeze(0)
        y = self.encoded[idx + 1 : idx + self.slice_len + 1].unsqueeze(0)
        return x.long(), y.long()
