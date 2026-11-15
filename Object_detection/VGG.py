import torch 
import torch.nn as nn

VGG16_ARCHITECTURE = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
class VGG16(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(VGG16, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_layers = self.create_conv_layer(VGG16_ARCHITECTURE)
        self.fcs= nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = nn.Flatten()(x)
        x = self.fcs(x)
        return x

    def create_conv_layer(self,architecture):
        layers = []
        in_channels = self.in_channels
        for v in architecture:
            if type(v) == int:
                out_channels = v
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
                           nn.BatchNorm2d(v),
                           nn.ReLU(inplace=True)]
                in_channels = v
            elif v == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
        return nn.Sequential(*layers)

