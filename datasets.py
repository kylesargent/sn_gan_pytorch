import os

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import chainer
import yaml

import sys

sys.path.insert(0, '/home/kyle/sngan_projection/source/')
sys.path.insert(0, '/home/kyle/sngan_projection/datasets/')
import yaml_utils

def get_dataset_struct(dataset, sn_gan_data_path, batch_size, num_workers, subsample=None):
    if dataset == "cifar10":
        return {
            'train_iter': get_cifar10_iter(os.path.join(sn_gan_data_path, 'cifar10/'), batch_size, num_workers, subsample),
            'fid_stats_path': os.path.join(sn_gan_data_path, 'cifar10/', 'fid_stats_cifar10_train.npz'),
            'n_classes': 10
        }
    else:
        raise NotImplementedError("Dataset loader not implemented")

    
def dunk(dataset_path, batch_size, num_workers, subsample):
    def cifar10_preprocess(tensor):
        transformed_tensor = (2. * tensor - 1.)
        # transformed_tensor += torch.rand(*transformed_tensor.shape) / 128.
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
        trainloader = torch.utils.data.DataLoader(
            trainset, 
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False
        )
    else:
        indices = np.random.choice(range(len(trainset)), size=int(len(trainset) * subsample))
        sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
        trainloader = torch.utils.data.DataLoader(
            trainset, 
            batch_size=batch_size,
            shuffle=False, 
            num_workers=num_workers,
            sampler=sampler
        )
    
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    train_iter = iter(cycle(trainloader))

    return train_iter

def get_cifar10_iter(dataset_path, batch_size, num_workers, subsample):
    config = yaml_utils.Config(yaml.load(open('/home/kyle/sngan_projection/configs/sn_cifar10_unconditional.yml')))
    dataset = yaml_utils.load_dataset(config)
    chainer_iterator = chainer.iterators.MultiprocessIterator(
            dataset, config.batchsize, n_processes=4, shuffle=False)

    def cycle(iterable):
        while True:
            for batch in iterable:
                xs = np.array([x for x, y in batch])
                yield torch.from_numpy(xs), None
    train_iter = iter(cycle(chainer_iterator))
    return train_iter