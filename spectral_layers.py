import torch
import torch.nn as nn
import torch.nn.functional as F

def max_singular_value(weight, u, Ip):
    assert(Ip >= 1)
    
    _u = u
    for _ in range(Ip):
        _v = F.normalize(torch.mm(_u, weight), p=2, dim=1).detach()
        _u = F.normalize(torch.mm(_v, weight.transpose(0,1)), p=2, dim=1).detach()
    sigma = torch.sum(F.linear(_u, weight.transpose(0,1)) * _v)
    return sigma, _u

def extended_singular_value(weight, u, Ip):
    assert(Ip >= 1)
    
    _u = u
    for _ in range(Ip):
        _v = F.normalize(torch.mm(_u, weight), p=2, dim=1).detach()
        _u = F.normalize(torch.mm(_v, weight.transpose(0,1)), p=2, dim=1).detach()
    sigma = torch.sum(F.linear(_u, weight.transpose(0,1)) * _v)
    return sigma, _u, _v

class SNLinear(nn.Linear):
    
    def __init__(self, in_features, out_features, bias=True, init_u=None, use_gamma=False):
        super(SNLinear, self).__init__(
            in_features, out_features, bias
        )
        self.Ip = 1
        self.register_buffer('u', init_u if init_u is not None else torch.randn(1, out_features))
        self.gamma = nn.Parameter(torch.zeros(1)) if use_gamma else None

    @property
    def W_bar(self):
        sigma, u = max_singular_value(self.weight, self.u, self.Ip)
        self.u[:] = u
        return self.weight / sigma

    def forward(self, x):
        if self.gamma is not None:
            return torch.exp(self.gamma) * F.linear(x, self.W_bar, self.bias) 
        else:
            return F.linear(x, self.W_bar, self.bias) 


class Linear_SpectralGradientClip(nn.Linear):
    
    def __init__(self, in_features, out_features, bias=True, init_u=None, use_gamma=False, r=1.2, Ip=4):
        super(Linear_SpectralGradientClip, self).__init__(
            in_features, out_features, bias
        )
        self.Ip = Ip
        self.r = r
        self.register_buffer('u0', init_u if init_u is not None else torch.randn(1, out_features))
        self.register_buffer('u1', init_u if init_u is not None else torch.randn(1, out_features))


    def clamp_gradient_spectra(self):  
        sigma0, u0, v0 = extended_singular_value(self.weight, self.u0, self.Ip)
        delta0 = sigma0 * torch.matmul(u0.transpose(0,1), v0)
        sigma1, u1, v1 = extended_singular_value(self.weight - delta0, self.u1, self.Ip)

        sigma_clamp = self.r * sigma1
        sigma0_scale = max(0, sigma0 - sigma_clamp)
        delta1 = sigma0_scale * torch.matmul(u0.transpose(0,1), v0)

        self.u0[:] = u0
        self.u1[:] = u1
        self.weight.grad -= delta1


class SNConv2d(nn.Conv2d):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, init_u=None, use_gamma=False):
        super(SNConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        self.Ip = 1
        self.register_buffer('u', init_u if init_u is not None else torch.randn(1, out_channels))
        self.gamma = nn.Parameter(torch.zeros(1)) if use_gamma else None


    @property
    def W_bar(self):
        w = self.weight
        w = w.view(w.shape[0], -1)
        
        sigma, u = max_singular_value(w, self.u, self.Ip)

        self.u[:] = u
        return self.weight / sigma
    
    def forward(self, x):
        r = F.conv2d(
            x, 
            self.W_bar,
            bias=self.bias,
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation,
            groups=self.groups
        )
        if self.gamma is not None:
            return torch.exp(self.gamma) * r
        else:
            return r


class Conv2d_SpectralGradientClip(nn.Conv2d):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, init_u=None, use_gamma=False, r=1.2, Ip=4):
        super(Conv2d_SpectralGradientClip, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        self.Ip = Ip
        self.r = r
        self.register_buffer('u0', init_u if init_u is not None else torch.randn(1, out_channels))
        self.register_buffer('u1', init_u if init_u is not None else torch.randn(1, out_channels))


    def clamp_gradient_spectra(self):
        w = self.weight.grad
        w = w.view(w.shape[0], -1)

        sigma0, u0, v0 = extended_singular_value(w, self.u0, self.Ip)
        delta0 = sigma0 * torch.matmul(u0.transpose(0,1), v0)
        sigma1, u1, v1 = extended_singular_value(w - delta0, self.u1, self.Ip)

        sigma_clamp = self.r * sigma1
        sigma0_scale = max(0, sigma0 - sigma_clamp)
        delta1 = sigma0_scale * torch.matmul(u0.transpose(0,1), v0)

        self.u0[:] = u0
        self.u1[:] = u1
        
        delta1 = delta1.view(*self.weight.grad.data.shape)
        self.weight.grad.data[:] -= delta1


class SNEmbedId(nn.Embedding): 

    def __init__(self, num_classes, num_features, init_u=None):
        super(SNEmbedId, self).__init__(
            num_classes, num_features
        )
        self.Ip = 1
        if init_u is not None:
            self.u = init_u
        else:
            self.u = torch.randn(1, num_features).to(device)

    @property
    def W_bar(self):
        sigma, u = max_singular_value(self.weight, self.u, self.Ip)
        self.u = u
        return self.weight / sigma

    def forward(self, x):
        return F.embedding(x, self.W_bar)
        
