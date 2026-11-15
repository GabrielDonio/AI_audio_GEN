import torch
import torchaudio
import os

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, root, max_len=16384, verbose=False):
        self.files = [os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith('.wav')]
        self.max_len = max_len
        self.verbose = verbose

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.files[idx])
        if sr != 16000:
            raise ValueError(f"Sample rate {sr} unexpected, should be 16000")

        if self.verbose:
            print(f"Chargement fichier {self.files[idx]} avec sr={sr}, shape={waveform.shape}")

        # Normalisation simple: diviser par la valeur absolue max du waveform pour avoir amplitude dans [-1, 1]
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val

        # Tronquer ou padder à max_len samples
        if waveform.shape[1] > self.max_len:
            waveform = waveform[:, :self.max_len]
            if self.verbose:
                print(f"  - Tronqué à {self.max_len}")
        elif waveform.shape[1] < self.max_len:
            padding = self.max_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            if self.verbose:
                print(f"  - Paddé à {self.max_len} (+{padding} zeros)")

        return waveform, 0
