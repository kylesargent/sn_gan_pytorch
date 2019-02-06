import torch.nn as nn
import torch.nn.functional as F
import torch
from resnet_blocks import GeneratorBlock, DiscriminatorBlock


class Generator(nn.Module):
    
    def __init__(self, z_size=128, bottom_width=4):
        super(Generator, self).__init__()
        
        self.bottom_width = bottom_width
        
        self.linear_1 = nn.Linear(z_size, (bottom_width ** 2) * 256)
        self.block_1 = GeneratorBlock(256, 256, upsample=True)
        self.block_2 = GeneratorBlock(256, 256, upsample=True)
        self.block_3 = GeneratorBlock(256, 256, upsample=True)
        self.batchnorm = nn.BatchNorm2d(256)
        self.conv = nn.Conv2d(256, 3, 3, padding=1)
        
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
    
class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.block1 = DiscriminatorBlock(3, 64, downsample=True)
        self.block2 = DiscriminatorBlock(64, 128, downsample=True)
        self.block3 = DiscriminatorBlock(128, 256, downsample=True)
        self.block4 = DiscriminatorBlock(256, 512, downsample=True)
        self.block5 = DiscriminatorBlock(512, 1024, downsample=False)
        
        self.dense = nn.Linear(1024, 1)
        
    def forward(self, x):
        p = self.block1(x)
        p = self.block2(p)
        p = self.block3(p)
        p = self.block4(p)
        p = self.block5(p)
        
        p = F.relu(p)
        p = torch.sum(p, dim=(2,3))
        
        p = self.dense(p)
        return p