import os
import torch
from datasets import get_dataset_struct
from cifar10_models import Cifar10Generator, Cifar10Discriminator

class TrainingWrapper(object):

    def __init__(self, d, g, d_optim, g_optim, d_scheduler, g_scheduler, config):
        self.d = d
        self.g = g
        self.d_optim = d_optim
        self.g_optim = g_optim
        self.config = config
        self.g_scheduler = g_scheduler
        self.d_scheduler = d_scheduler

    def save(self, path):
        state = {
            'd': self.d.state_dict(),
            'g': self.g.state_dict(),
            'd_optim': self.d_optim.state_dict(),
            'g_optim': self.g_optim.state_dict(),
            'd_scheduler': self.d_scheduler.state_dict(),
            'g_scheduler': self.g_scheduler.state_dict(),
            'config': self.config
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        state = torch.load(path, map_location=device)
        config = state['config']

        # some hacks for backward compat
        config['n_classes'] = 0
        config['eval_batch_size'] = 512

        d = Cifar10Discriminator(n_classes=config['n_classes'])
        g = Cifar10Generator(n_classes=config['n_classes'])
        d.load_state_dict(state['d'])
        g.load_state_dict(state['g'])

        d_optim = torch.optim.Adam(d.parameters(), lr=.0002)
        g_optim = torch.optim.Adam(g.parameters(), lr=.0002)
        d_optim.load_state_dict(state['d_optim'])
        g_optim.load_state_dict(state['g_optim'])

        linear_decay = lambda iteration: 1 - (iteration / config['max_iters'])
        d_scheduler = torch.optim.lr_scheduler.LambdaLR(d_optim, linear_decay, -1)
        g_scheduler = torch.optim.lr_scheduler.LambdaLR(g_optim, linear_decay, -1)

        return TrainingWrapper(d, g, d_optim, g_optim, d_scheduler, g_scheduler, config)
