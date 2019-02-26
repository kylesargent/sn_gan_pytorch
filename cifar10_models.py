import torch.nn as nn
import torch.nn.functional as F
import torch
from resnet_layers import GeneratorBlock, DiscriminatorBlock
from torch.nn.init import xavier_uniform_
from torch.nn.utils import spectral_norm


class Cifar10Generator(nn.Module):
    
    def __init__(self, z_size=128, bottom_width=4, n_classes=0):
        super(Cifar10Generator, self).__init__()
        
        self.bottom_width = bottom_width
        
        self.linear_1 = nn.Linear(z_size, (bottom_width ** 2) * 256)
        self.block_1 = GeneratorBlock(256, 256, upsample=True, n_classes=n_classes)
        self.block_2 = GeneratorBlock(256, 256, upsample=True, n_classes=n_classes)
        self.block_3 = GeneratorBlock(256, 256, upsample=True, n_classes=n_classes)
        self.batchnorm = nn.BatchNorm2d(256)
        self.conv = nn.Conv2d(256, 3, 3, padding=1)

        xavier_uniform_(self.linear_1.weight)
        xavier_uniform_(self.conv.weight)

        
    def forward(self, z, y=None):
        if y is not None:
            assert(z.shape[0] == y.shape[0])

        x = self.linear_1(z)
        x = x.view(x.shape[0], -1, self.bottom_width, self.bottom_width)
        
        x = self.block_1(x, y)
        x = self.block_2(x, y)
        x = self.block_3(x, y)
        x = self.batchnorm(x)
        x = F.relu(x)
        x = self.conv(x)
        x = torch.tanh(x)
        return x
    
class Cifar10Discriminator(nn.Module):
    
    def __init__(self, channels=128, n_classes=0):
        super(Cifar10Discriminator, self).__init__()
        
        self.block1 = DiscriminatorBlock(3, channels, downsample=True, optimized=True)
        self.block2 = DiscriminatorBlock(channels, channels, downsample=True)
        self.block3 = DiscriminatorBlock(channels, channels, downsample=False)
        self.block4 = DiscriminatorBlock(channels, channels, downsample=False)
        
        self.dense = spectral_norm(nn.Linear(channels, 1, bias=False))
        xavier_uniform_(self.dense.weight)

        if n_classes > 0:
            self.class_embedding = spectral_norm(nn.Embedding(n_classes, channels))
            xavier_uniform_(self.class_embedding.weight)
        
    def forward(self, x, y=None):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = F.relu(h)

        h = torch.sum(h, dim=(2,3))
        output = self.dense(h)

        if y is not None:
            assert(len(y.shape) == 1)

            label_weights = self.class_embedding(y)
            output += torch.sum(label_weights * h, dim=1, keepdim=True)

        return output