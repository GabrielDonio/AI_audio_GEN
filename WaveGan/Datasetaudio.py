import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, folder, slice_len=16384, sample_rate=16000):
        super().__init__()
        self.slice_len = slice_len
        self.sample_rate = sample_rate

        if isinstance(folder, str) and os.path.isdir(folder):
            # Chemins complets et triés
            self.files = sorted(glob.glob(os.path.join(folder, '*.wav')))
        elif isinstance(folder, (list, tuple)):
            # Liste explicite de chemins
            self.files = list(folder)
        else:
            raise ValueError(f"Invalid folder: {folder}")

        if not self.files:
            raise RuntimeError(f"No .wav files found in {folder}")

        print(f"✅ Found {len(self.files)} audio files.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            # charger en mono, resampler si besoin
            waveform, sr = librosa.load(path, sr=self.sample_rate, mono=True)
            waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)  # [1, T]
        except Exception as e:
            raise RuntimeError(f"Failed to load {path}: {e}")

        current_len = waveform.shape[1]
        if current_len >= self.slice_len:
            waveform = waveform[:, :self.slice_len]
        else:
            pad = self.slice_len - current_len
            waveform = F.pad(waveform, (0, pad))

        return waveform
