import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from spectral_layers import SNConv2d
import torch

class ConditionalBatchNorm2d(nn.Module):
    # Thanks to https://github.com/Kaixhin for this layer

    def __init__(self, num_features, num_classes, use_gamma=False):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features] = 1  
        self.embed.weight.data[:, num_features:] = 0  
        self.use_gamma = use_gamma

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        if self.use_gamma:
            out = gamma.view(-1, self.num_features, 1, 1) * out
        out += beta.view(-1, self.num_features, 1, 1)
        return out


class GeneratorBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, hidden_channels=None, kernel_size=3, padding=1, activation=F.relu, upsample=False, n_classes=0):
        super(GeneratorBlock, self).__init__()
        
        self.activation = activation
        self.upsample = upsample
        self.n_classes = n_classes
            
        self.learnable_shortcut = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_size, padding=padding)

        if self.n_classes == 0:
            self.b1 = nn.BatchNorm2d(in_channels, eps=2e-5)
            self.b2 = nn.BatchNorm2d(hidden_channels, eps=2e-5)
        else:
            self.b1 = ConditionalBatchNorm2d(in_channels, self.n_classes)
            self.b2 = ConditionalBatchNorm2d(hidden_channels, self.n_classes)
        
        if self.learnable_shortcut:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            xavier_uniform_(self.shortcut.weight)

        xavier_uniform_(self.conv1.weight, gain=1.41)
        xavier_uniform_(self.conv2.weight, gain=1.41)
            
    def forward(self, x, y=None):
        # residual
        assert((y is None and self.n_classes == 0) or (y is not None and self.n_classes > 0))

        r = self.b1(x) if y is None else self.b1(x, y)
        r = self.activation(r)
        if self.upsample:
            r = F.interpolate(r, scale_factor=2)
        r = self.conv1(r)
        r = self.b2(r) if y is None else self.b2(r, y)
        r = self.activation(r)
        r = self.conv2(r)
        
        # shortcut
        x_sc = x
        if self.learnable_shortcut:
            if self.upsample:
                x_sc = F.interpolate(x_sc, scale_factor=2)
            x_sc = self.shortcut(x_sc)
            
        return r + x_sc

class SNGeneratorBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, hidden_channels=None, kernel_size=3, padding=1, activation=F.relu, upsample=False, n_classes=0):
        super(SNGeneratorBlock, self).__init__()
        
        self.activation = activation
        self.upsample = upsample
        self.n_classes = n_classes
            
        self.learnable_shortcut = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        
        self.conv1 = SNConv2d(SNConv2d, hidden_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = SNConv2d(hidden_channels, out_channels, kernel_size=kernel_size, padding=padding)

        if self.n_classes == 0:
            self.b1 = nn.BatchNorm2d(in_channels, eps=2e-5)
            self.b2 = nn.BatchNorm2d(hidden_channels, eps=2e-5)
        else:
            self.b1 = ConditionalBatchNorm2d(in_channels, self.n_classes)
            self.b2 = ConditionalBatchNorm2d(hidden_channels, self.n_classes)
        
        if self.learnable_shortcut:
            self.shortcut = SNConv2d(in_channels, out_channels, kernel_size=1)
            xavier_uniform_(self.shortcut.weight)

        xavier_uniform_(self.conv1.weight, gain=1.41)
        xavier_uniform_(self.conv2.weight, gain=1.41)
            
    def forward(self, x, y=None):
        # residual
        assert((y is None and self.n_classes == 0) or (y is not None and self.n_classes > 0))

        r = self.b1(x) if y is None else self.b1(x, y)
        r = self.activation(r)
        if self.upsample:
            r = F.interpolate(r, scale_factor=2)
        r = self.conv1(r)
        r = self.b2(r) if y is None else self.b2(r, y)
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
    
    def __init__(self, in_channels, out_channels, hidden_channels=None, kernel_size=3, padding=1, activation=F.relu, downsample=False, optimized=False, use_gamma=False):
        super(DiscriminatorBlock, self).__init__()
        
        self.activation = activation
        self.downsample = downsample
        self.optimized = optimized
            
        self.learnable_shortcut = (in_channels != out_channels) or downsample
        if hidden_channels is None:
            if optimized:
                hidden_channels = out_channels
            else:
                hidden_channels = in_channels
        
        self.conv1 = SNConv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding, use_gamma=use_gamma)
        self.conv2 = SNConv2d(hidden_channels, out_channels, kernel_size=kernel_size, padding=padding, use_gamma=use_gamma)
        
        if self.learnable_shortcut:
            self.shortcut = SNConv2d(in_channels, out_channels, kernel_size=1, use_gamma=use_gamma)
            xavier_uniform_(self.shortcut.weight)

        xavier_uniform_(self.conv1.weight, gain=1.41)
        xavier_uniform_(self.conv2.weight, gain=1.41)

        
    def get_residual(self, x):
        r = x
        if not self.optimized:
            r = self.activation(r)
        r = self.conv1(r)
        r = self.activation(r)
        r = self.conv2(r)
        if self.downsample:
            r = F.avg_pool2d(r, 2)
        return r

    def get_shortcut(self, x):
        x_sc = x
        if self.optimized:
            return self.shortcut(F.avg_pool2d(x_sc, 2))
        else:
            if self.learnable_shortcut:
                x_sc = self.shortcut(x_sc)
                if self.downsample:
                    x_sc = F.avg_pool2d(x_sc, 2)
        return x_sc

    def forward(self, x):
        return self.get_residual(x) + self.get_shortcut(x)
