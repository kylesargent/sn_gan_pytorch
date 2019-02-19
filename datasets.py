import os

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

def get_dataset_struct(dataset, sn_gan_data_path, batch_size, subsample=None):
	if dataset == "cifar10":
		return {
			'train_iter': get_cifar10_iter(os.path.join(sn_gan_data_path, 'cifar10/'), batch_size, subsample),
			'fid_stats_path': os.path.join(sn_gan_data_path, 'cifar10/', 'fid_stats_cifar10_train.npz')
		}
	else:
		raise NotImplementedError("Dataset loader not implemented")

    
def get_cifar10_iter(dataset_path, batch_size, subsample):
	def cifar10_preprocess(tensor):
	    transformed_tensor = 2. * tensor - 1.
	    transformed_tensor += torch.rand(*transformed_tensor.shape) / 128.
	    return transformed_tensor

	transform = transforms.Compose(
	    [
	        transforms.ToTensor(),
	        cifar10_preprocess
	    ]
	)
	trainset = torchvision.datasets.CIFAR10(
		root=dataset_path, 
		train=True,
		download=True, 
		transform=transform
	)

	if subsample == None:
		sampler = None
	else:
		indices = np.random.choice(range(len(trainset)), size=int(len(trainset) * subsample))
		sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)

	trainloader = torch.utils.data.DataLoader(
		trainset, 
		batch_size=batch_size,
		shuffle=False, 
		num_workers=1,
		sampler=sampler
	)
	return trainloader