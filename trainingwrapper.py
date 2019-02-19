import os
import torch
from datasets import get_dataset_struct
from cifar10_models import Cifar10Generator, Cifar10Discriminator

class TrainingWrapper(object):

	def __init__(self, d, g, d_optim, g_optim, config):
		# epoch, optims, g, d, train_iter, device, args
		self.d = d
		self.g = g
		self.d_optim = d_optim
		self.g_optim = g_optim
		self.config = config

	def save(self, path):
		state = {
			'd': self.d.state_dict(),
			'g': self.g.state_dict(),
			'd_optim': self.d_optim.state_dict(),
			'g_optim': self.g_optim.state_dict(),
			'config': self.config
		}
		torch.save(state, path)

	@classmethod
	def load(cls, path):
		state = torch.load(path)
		config = state['config']

		d = Cifar10Discriminator()
		g = Cifar10Generator()
		d.load_state_dict(state['d'])
		g.load_state_dict(state['g'])

		d_optim = torch.optim.Adam(d.parameters(), lr=.0002)
		g_optim = torch.optim.Adam(g.parameters(), lr=.0002)

		return TrainingWrapper(d, g, d_optim, g_optim, config)