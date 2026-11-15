import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from WaveNet.WavenetModel import WaveNetModel
from WaveNet.Dataset import WaveNetDataset as Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Hyperparameters
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
INPUTSIZE = 16384
NUM_LAYERS = 10
KERNEL_SIZE = 2

model = WaveNetModel(in_channels=1, res_channels=64, skip_channels=64, out_channels=256, num_layers=NUM_LAYERS, kernel_size=KERNEL_SIZE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

 #transforms = transforms.Compose()
files = "A:/Program Files/Documents infaudio/AI_audio/WaveGan/drums/train"
dataset = Dataset(files, slice_len=INPUTSIZE, sample_rate=16000)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)

model.train()
def train_step(data, target):
    data, target = data.to(device), target.to(device).long().squeeze(1)
    optimizer.zero_grad()
    output = model(data)  # output shape: [B, 256, T]
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(EPOCHS):
    for batch in loader:  
        data, target = batch
        loss = train_step(data, target)
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss:.4f}')

