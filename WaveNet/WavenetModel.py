import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(DilatedConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.padding] if self.padding != 0 else out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, res_channels, skip_channels, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        self.filter_conv = DilatedConv1d(in_channels, res_channels, kernel_size, dilation)
        self.gate_conv = DilatedConv1d(in_channels, res_channels, kernel_size, dilation)

        self.residual_conv = nn.Conv1d(res_channels, in_channels, 1)
        self.skip_conv = nn.Conv1d(res_channels, skip_channels, 1)

    def forward(self, x):
        filter_out = torch.tanh(self.filter_conv(x))
        gate_out = torch.sigmoid(self.gate_conv(x))
        z = filter_out * gate_out

        skip = self.skip_conv(z)
        residual = self.residual_conv(z)
        output = x + residual
        return output, skip


class WaveNetModel(nn.Module):
    def __init__(self, in_channels, res_channels, skip_channels, out_channels, num_layers, kernel_size):
        super(WaveNetModel, self).__init__()
        self.input_conv = nn.Conv1d(in_channels, res_channels, kernel_size=1)

        self.res_blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.res_blocks.append(
                ResidualBlock(res_channels, res_channels, skip_channels, kernel_size, dilation)
            )

        self.relu = nn.ReLU()
        self.output_conv1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.output_conv2 = nn.Conv1d(skip_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.input_conv(x)
        skip_connections = []

        for block in self.res_blocks:
            x, skip = block(x)
            skip_connections.append(skip)

        x = sum(skip_connections)
        x = self.relu(x)
        x = self.output_conv1(x)
        x = self.relu(x)
        x = self.output_conv2(x)
        x = self.softmax(x) 

        return x
    
if __name__ == "__main__":
    batchSize = 4
    inChannels = 1
    inputLength = 16000

    noise = torch.randn(batchSize, inChannels, inputLength)
    model = WaveNetModel(in_channels=inChannels, res_channels=64, skip_channels=64, out_channels=256, num_layers=10, kernel_size=2)
    output = model(noise)
    print("Output shape:", output.shape)  # Should be (BatchSize, out_channels, InputLength)
