import torch.nn as nn
import torch.nn.functional as F

class GeneratorBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, hidden_channels=None, kernel_size=3, padding=1, activation=F.relu, upsample=False):
        super(GeneratorBlock, self).__init__()
        
        self.activation = activation
        self.upsample = upsample
            
        self.learnable_shortcut = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_size, padding=padding)
        
        self.b1 = nn.BatchNorm2d(in_channels)
        self.b2 = nn.BatchNorm2d(hidden_channels)
        
        if self.learnable_shortcut:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            
    def forward(self, x):
        # residual
        r = self.b1(x)
        r = self.activation(r)
        if self.upsample:
            r = F.interpolate(r, scale_factor=2)
        r = self.conv1(r)
        r = self.b2(r)
        r = self.activation(r)
        r = self.conv2(r)
        
        # shortcut
        x_sc = x
        if self.learnable_shortcut:
            if self.upsample:
                x_sc = F.interpolate(x_sc, scale_factor=2)
            x_sc = self.shortcut(x_sc)
            
        return r + x_sc

class DiscriminatorBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, hidden_channels=None, kernel_size=3, padding=1, activation=F.relu, downsample=False):
        super(DiscriminatorBlock, self).__init__()
        
        self.activation = activation
        self.downsample = downsample
            
        self.learnable_shortcut = in_channels != out_channels or downsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_size, padding=padding)
        
        if self.learnable_shortcut:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            
    def forward(self, x):
        # residual
        r = self.activation(x)
        r = self.conv1(r)
        r = self.activation(r)
        r = self.conv2(r)
        if self.downsample:
            r = F.avg_pool2d(r, 2)
        
        # shortcut
        x_sc = x
        if self.learnable_shortcut:
            x_sc = self.shortcut(x_sc)
            if self.downsample:
                x_sc = F.avg_pool2d(x_sc, 2)
            
        return r + x_sc

