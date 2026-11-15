import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from CNN_ex import SimpleCNN
from LeNet import LeNet
from VGG import VGG16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Hyperparameters
input_size = 784  # Example input size for MNIST
in_channels = 1  # Number of input channels for grayscale images
num_classes = 10  # Number of classes in MNIST
learning_rate = 0.001
num_epoch=10
BATCH_SIZE = 64

train_dataset = datasets.MNIST(root='Object_detection/data', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = datasets.MNIST(root='Object_detection/data', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model= VGG16(in_channels=1, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epoch):
    for batch_idx, (data,targets) in enumerate(train_loader):
        data=data.to(device)
        targets=targets.to(device)

        # Reshape the data to match the input size of the model
        #data = data.reshape((data.size(0), -1)) 

        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epoch}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
def check_accuracy(loader, model):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

check_accuracy(test_loader, model)
check_accuracy(train_loader, model)
