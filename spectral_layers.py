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
    return sigma, _u, _v

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]

class SNLinear(nn.Linear):
    
    def __init__(self, in_features, out_features, bias=True, init_u=None, use_gamma=False):
        super(SNLinear, self).__init__(
            in_features, out_features, bias
        )
        self.Ip = 1
        self.register_buffer('u', init_u if init_u is not None else torch.randn(1, out_features))
        self.gamma = nn.Parameter(torch.zeros(1)) if use_gamma else None

        self.Ip_grad = 8
        self.r = 10
        self.register_buffer('u0', init_u if init_u is not None else torch.randn(1, out_features))
        self.register_buffer('u1', init_u if init_u is not None else torch.randn(1, out_features))

    @property
    def W_bar(self):
        sigma, u, _ = max_singular_value(self.weight, self.u, self.Ip)
        self.u[:] = u
        return self.weight / sigma

    def clamp_gradient_spectra(self):
        if self.weight.shape[0] > 1:  
            sigma0, u0, v0 = extended_singular_value(self.weight.grad, self.u0, self.Ip_grad)
            delta = torch.matmul(u0.transpose(0,1), v0)

            delta0 = sigma0 * delta
            sigma1, u1, v1 = extended_singular_value(self.weight.grad - delta0, self.u1, self.Ip_grad)

            sigma_clamp = self.r * sigma1
            sigma0_scale = max(0, sigma0 - sigma_clamp)
            delta1 = sigma0_scale * delta

            self.u0[:] = u0
            self.u1[:] = u1
            self.weight.grad -= delta1

    def forward(self, x):
        if self.gamma is not None:
            return torch.exp(self.gamma) * F.linear(x, self.W_bar, self.bias) 
        else:
            return F.linear(x, self.W_bar, self.bias) 


class SNConv2d(nn.Conv2d):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, init_u=None, use_gamma=False):
        super(SNConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        self.Ip = 1
        
        self.register_buffer('u', init_u if init_u is not None else torch.randn(1, out_channels))
        self.gamma = nn.Parameter(torch.zeros(1)) if use_gamma else None

        self.Ip_grad = 8
        self.r = 10
        self.register_buffer('u0', init_u if init_u is not None else torch.randn(1, out_channels))
        self.register_buffer('u1', init_u if init_u is not None else torch.randn(1, out_channels))

    @property
    def W_bar(self):
        w = self.weight
        w = w.view(w.shape[0], -1)
        
        sigma, u, _ = max_singular_value(w, self.u, self.Ip)
        self.u[:] = u
        return self.weight / sigma

    def sigma(self):
        w = self.weight
        w = w.view(w.shape[0], -1)
        sigma, _, _ = max_singular_value(w, self.u, self.Ip)
        return sigma
    
    def clamp_gradient_spectra(self):
        if self.weight.shape[0] > 1:  
            w = self.weight.grad
            w = w.view(w.shape[0], -1)

            sigma0, u0, v0 = extended_singular_value(w, self.u0, self.Ip_grad)
            delta = torch.matmul(u0.transpose(0,1), v0)

            delta0 = sigma0 * delta
            sigma1, u1, v1 = extended_singular_value(w - delta0, self.u1, self.Ip_grad)

            sigma_clamp = self.r * sigma1
            sigma0_scale = max(0, sigma0 - sigma_clamp)
            delta1 = sigma0_scale * delta

            self.u0[:] = u0
            self.u1[:] = u1
            self.weight.grad -= delta1.view(*self.weight.grad.shape)

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

class SNConv2dToeplitz(nn.Conv2d):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, init_u=None, use_gamma=False):
        super(SNConv2dToeplitz, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        self.Ip = 1
        self.gamma = nn.Parameter(torch.zeros(1)) if use_gamma else None
        self.has_u = False

    def max_singular_value_toeplitz(self, u, Ip=1):
        transpose_weight = flip(flip(self.weight.transpose(0,1), 2), 3)

        # no bias necessary for SN computation, and we already padded in the first pass for computing u
        W = lambda input: F.conv2d(input, self.weight, bias=None, stride=1, padding=0)
        Wt = lambda input: F.conv2d(input, transpose_weight, bias=None, stride=1, padding=self.kernel_size[0] - 1)

        with torch.no_grad():
            for _ in range(Ip):
                _v = Wt(u)
                _v = (_v / torch.norm(_v)).detach()

                _u = W(_v)
                _u = (_u / torch.norm(_u)).detach()
        sigma = torch.dot(_u.view(-1), W(_v).view(-1))
        return sigma, _u

    @property
    def W_bar(self):
        sigma, u = self.max_singular_value_toeplitz(self.u, self.Ip)
        self.u[:] = u
        return self.weight / sigma

    def sigma(self):
        sigma, u = self.max_singular_value_toeplitz(self.u, self.Ip)
        self.u[:] = u
        return sigma

    def forward(self, x):
        if not self.has_u:
            self.register_buffer('u', torch.randn_like(super().forward(x)))
            self.has_u = True

        return 4 * F.conv2d(
            x, 
            self.W_bar,
            bias=self.bias,
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation,
            groups=self.groups
        )


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
        sigma, u, _ = max_singular_value(self.weight, self.u, self.Ip)
        self.u = u
        return self.weight / sigma

    def forward(self, x):
        return F.embedding(x, self.W_bar)
        
