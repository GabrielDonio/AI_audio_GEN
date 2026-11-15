import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

class Visualisation:
    def __init__(self, log_dir="runs/experiment"):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 0

    def normalize(self, waveform):
        max_val = waveform.abs().max(dim=1, keepdim=True)[0]
        max_val[max_val == 0] = 1.0
        return waveform / max_val

    def log_waveforms(self, real_waveform, generated_waveform, 
                      tag_real="Real", tag_generated="Generated", sample_rate=16000, max_samples=4):
        # Si 1D, on ajoute la batch dim
        if real_waveform.dim() == 1:
            real_waveform = real_waveform.unsqueeze(0)
        if generated_waveform.dim() == 1:
            generated_waveform = generated_waveform.unsqueeze(0)
        # Retirer la dimension channel (si elle existe)
        if real_waveform.dim() == 3 and real_waveform.size(1) == 1:
            real_waveform = real_waveform.squeeze(1)
        if generated_waveform.dim() == 3 and generated_waveform.size(1) == 1:
            generated_waveform = generated_waveform.squeeze(1)

        real_waveform = self.normalize(real_waveform)
        generated_waveform = self.normalize(generated_waveform)

        n_samples = min(real_waveform.size(0), generated_waveform.size(0), max_samples)
        for i in range(n_samples):
            self.writer.add_audio(f"{tag_real}/sample_{i}", real_waveform[i].unsqueeze(0), self.step, sample_rate=sample_rate)
            self.writer.add_audio(f"{tag_generated}/sample_{i}", generated_waveform[i].unsqueeze(0), self.step, sample_rate=sample_rate)
            # Figure forme d'onde
            self._log_waveform_plot(real_waveform[i], f"{tag_real}_waveform/sample_{i}")
            self._log_waveform_plot(generated_waveform[i], f"{tag_generated}_waveform/sample_{i}")

        self.step += 1

    def _log_waveform_plot(self, waveform, tag):
        waveform_np = waveform.detach().cpu().squeeze().numpy()
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.plot(waveform_np)
        ax.set_title(tag)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Amplitude")
        ax.grid(True)

        self.writer.add_figure(tag, fig, global_step=self.step)
        plt.close(fig)
    def log_weights(self, model, tag_prefix="Model", step=None):
        """
        Log histograms of model weights for TensorBoard.
        """
        current_step = step if step is not None else self.step
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f"{tag_prefix}/{name}", param, global_step=current_step)


    def close(self):
        self.writer.close()
