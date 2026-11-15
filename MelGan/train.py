import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from Model import Generator, MultiScaleDiscriminator  
from audioDataset import AudioDataset  

def hinge_loss_discriminator(real_preds, fake_preds):
    loss_real = torch.mean(F.relu(1.0 - real_preds))
    loss_fake = torch.mean(F.relu(1.0 + fake_preds))
    return loss_real + loss_fake

def hinge_loss_generator(fake_preds):
    return -torch.mean(fake_preds)

def feature_matching_loss(real_features, fake_features):
    loss = 0
    for real_fmap, fake_fmap in zip(real_features, fake_features):
        for real, fake in zip(real_fmap, fake_fmap):
            min_len = min(real.size(-1), fake.size(-1))
            real_trim = real[..., :min_len]
            fake_trim = fake[..., :min_len]
            loss += torch.mean(torch.abs(real_trim - fake_trim))
    return loss


def train(audio_dir, epochs=7, batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Active la détection d'anomalies PyTorch pour debugger
    torch.autograd.set_detect_anomaly(True)
    
    dataset = AudioDataset(audio_dir, n_mels=120)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    gen = Generator(n_mels=120).to(device)
    disc = MultiScaleDiscriminator().to(device)

    optim_g = optim.Adam(gen.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optim_d = optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.9))

    for epoch in range(epochs):
        for i, (mel, real_audio) in enumerate(dataloader):
            mel = mel.to(device)
            real_audio = real_audio.to(device)

            # Générateur -> audio fake
            fake_audio = gen(mel)

            # Discriminateur sur vrai et fake audio
            # Ajout de .unsqueeze(1) pour fake_audio pour respecter la forme (B, 1, L)
            real_preds, real_feats = disc(real_audio.unsqueeze(1))
            fake_preds, fake_feats = disc(fake_audio.detach().unsqueeze(1))

            # Perte discriminateur sans retain_graph=True
            loss_d = hinge_loss_discriminator(real_preds[-1], fake_preds[-1])
            optim_d.zero_grad()
            loss_d.backward()  # <- plus de retain_graph ici !
            optim_d.step()

            # Perte générateur (adv + feature matching)
            # De même, unsqueeze ici pour correspondre à la forme attendue
            fake_preds, fake_feats = disc(fake_audio.unsqueeze(1))
            loss_g_adv = hinge_loss_generator(fake_preds[-1])
            loss_fm = feature_matching_loss(real_feats, fake_feats) * 10.0
            loss_g = loss_g_adv + loss_fm

            optim_g.zero_grad()
            loss_g.backward()
            optim_g.step()

            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Step [{i}/{len(dataloader)}] "
                      f"Loss D: {loss_d.item():.4f} Loss G: {loss_g.item():.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(gen.state_dict(), f"generator_epoch_{epoch+1}.pt")
            torch.save(disc.state_dict(), f"discriminator_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    audio_dir = r"A:\Program Files\Documents infaudio\AI_audio\WaveGan\drums\train"
    train(audio_dir)
