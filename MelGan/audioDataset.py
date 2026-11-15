import torch
import torchaudio
from torch.utils.data import Dataset
import os

class AudioDataset(Dataset):
    def __init__(self, audio_dir, sample_rate=16000, segment_length=16384, n_mels=80, n_fft=1024, hop_length=256):
        """
        audio_dir: dossier contenant les fichiers audio (.wav, etc)
        sample_rate: taux d'échantillonnage attendu
        segment_length: longueur en échantillons de la tranche audio (ici 1 sec)
        n_mels: nombre de bandes Mel
        n_fft, hop_length: paramètres STFT pour spectrogramme
        """
        super().__init__()
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length

        # Liste des fichiers audio
        self.audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".wav")]

        # MelSpectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=1.0  # amplitude spectrogram
        )

        # Normalisation log (optionnel)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # Chargement du fichier audio
        waveform, sr = torchaudio.load(self.audio_files[idx])

        # Si stereo, convertir en mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample si nécessaire
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)

        # Découper ou padding pour avoir segment_length
        if waveform.size(1) >= self.segment_length:
            start = torch.randint(0, waveform.size(1) - self.segment_length + 1, (1,)).item()
            waveform = waveform[:, start:start + self.segment_length]
        else:
            # Padding zéro si plus court
            pad_len = self.segment_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        # Calcul Mel-spectrogramme
        mel = self.mel_transform(waveform)  # (1, n_mels, time_frames)

        # Optionnel : passer en dB (log) pour meilleure stabilité
        mel = self.amplitude_to_db(mel)

        # Normalisation (par exemple min-max ou standard) peut être ajouté ici

        return mel.squeeze(0), waveform.squeeze(0)
