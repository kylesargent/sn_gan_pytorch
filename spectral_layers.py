import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def max_singular_value(weight, u, Ip):
    assert(Ip >= 1)
    
    _u = u
    for _ in range(Ip):
        _v = F.normalize(torch.mm(_u, weight), p=2, dim=1).detach()
        _u = F.normalize(torch.mm(_v, weight.transpose(0,1)), p=2, dim=1).detach()
    sigma = torch.sum(F.linear(_u, weight.transpose(0,1)) * _v)
    return sigma, _u

class SNLinear(nn.Linear):
    
    def __init__(self, in_features, out_features, bias=True, init_u = None, use_gamma=False):
        super(SNLinear, self).__init__(
            in_features, out_features, bias
        )
        self.Ip = 1
        if init_u is not None:
            self.u = init_u
        else:
            self.u = torch.randn(1, out_features).to(device)
        self.use_gamma = use_gamma
        if self.use_gamma:
            self.gamma = nn.Parameter(torch.ones(1))

    @property
    def W_bar(self):
        sigma, u = max_singular_value(self.weight, self.u, self.Ip)
        self.u = u
        return self.weight / sigma

    def forward(self, x):
        return torch.exp(self.gamma) * F.linear(x, self.W_bar, self.bias)
    
class SNConv2d(nn.Conv2d):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, init_u=None, use_gamma=False):
        super(SNConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        self.Ip = 1
        if init_u is not None:
            self.u = init_u
        else:
            self.u = torch.randn(1, out_channels).to(device)
        
        self.use_gamma = use_gamma
        if self.use_gamma:
            self.gamma = nn.Parameter(torch.ones(1))


    @property
    def W_bar(self):
        w = self.weight
        w = w.view(w.shape[0], -1)
        
        sigma, u = max_singular_value(w, self.u, self.Ip)

        self.u = u
        return self.weight / sigma
    
    def forward(self, x):
        return torch.exp(self.gamma) * F.conv2d(
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
        sigma, u = max_singular_value(self.weight, self.u, self.Ip)
        self.u = u
        return self.weight / sigma

    def forward(self, x):
        return F.embedding(x, self.W_bar)
        