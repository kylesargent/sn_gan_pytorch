import torch.nn as nn
import torch.nn.functional as F
import torch
from resnet_blocks import GeneratorBlock, DiscriminatorBlock
from torch.nn.init import xavier_uniform_
from torch.nn.utils import spectral_norm


class Cifar10Generator(nn.Module):
    
    def __init__(self, z_size=128, bottom_width=4):
        super(Cifar10Generator, self).__init__()
        
        self.bottom_width = bottom_width
        
        self.linear_1 = nn.Linear(z_size, (bottom_width ** 2) * 256)
        self.block_1 = GeneratorBlock(256, 256, upsample=True)
        self.block_2 = GeneratorBlock(256, 256, upsample=True)
        self.block_3 = GeneratorBlock(256, 256, upsample=True)
        self.batchnorm = nn.BatchNorm2d(256)
        self.conv = nn.Conv2d(256, 3, 3, padding=1)

        xavier_uniform_(self.linear_1.weight)
        xavier_uniform_(self.conv.weight)

        
    def forward(self, z):
        x = self.linear_1(z)
        x = x.view(x.shape[0], -1, self.bottom_width, self.bottom_width)
        
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        x = self.conv(x)
        x = torch.tanh(x)
        return x
    
class Cifar10Discriminator(nn.Module):
    
    def __init__(self):
        super(Cifar10Discriminator, self).__init__()
        
        self.block1 = DiscriminatorBlock(3, 128, downsample=True, optimized=True)
        self.block2 = DiscriminatorBlock(128, 128, downsample=True)
        self.block3 = DiscriminatorBlock(128, 128, downsample=False)
        self.block4 = DiscriminatorBlock(128, 128, downsample=False)
        
        self.dense = spectral_norm(nn.Linear(128, 1, bias=False))
        xavier_uniform_(self.dense.weight)
        
    def forward(self, x):
        p = self.block1(x)
        p = self.block2(p)
        p = self.block3(p)
        p = self.block4(p)
        p = F.relu(p)

        p = torch.sum(p, dim=(2,3))
        
        p = self.dense(p)
        return p