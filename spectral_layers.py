import torch.nn as nn
import torch.nn.functional as F

def max_singular_value(weight, u, Ip):
    assert(Ip >= 1)
    
    _u = u
    for _ in range(Ip):
        _v = F.normalize(torch.mm(_u, weight), p=2, dim=1)
        _u = F.normalize(torch.mm(_v, weight.transpose(0,1)), p=2, dim=1)
    sigma = torch.sum(F.linear(_u, weight.transpose(0,1)) * _v)
    return sigma, _u

class SNLinear(nn.Linear):
    
    def __init__(self, in_features, out_features):
        super(SNLinear, self).__init__(
            in_features, out_features
        )
        self.Ip = 1
        self.u = torch.randn(1, out_features)
        
    @property
    def W_bar(self):
        sigma, u = max_singular_value(self.weight.data, self.u, self.Ip)
        self.u = u
        return self.weight / sigma

    def forward(self, x):
        return F.linear(x, self.W_bar, self.bias)
    
class SNConv2d(nn.Conv2d):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(SNConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        self.Ip = 1
        self.u = torch.randn(1, out_channels)
        
    @property
    def W_bar(self):
        w = self.weight.data
        w = w.view(w.shape[0], -1)
        
        sigma, u = max_singular_value(w, self.u, self.Ip)
        self.u = u
        return self.weight / sigma
    
    def forward(self, x):
        return F.conv2d(
            x, 
            self.W_bar,
            bias=self.bias,
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation,
            groups=self.groups
        )