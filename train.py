import argparse
import os
from os.path import expanduser
import logging
from time import gmtime, strftime

from cifar10_models import Cifar10Generator, Cifar10Discriminator
from trainingwrapper import TrainingWrapper
from update import update

import torch

DEFAULT_SN_GAN_DATA_PATH = os.path.expanduser('~/sn_gan_pytorch_data')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sn_gan_data_path', type=str, default=DEFAULT_SN_GAN_DATA_PATH, help='where to put model data and downloads')
    parser.add_argument('--dataset', type=str, default='cifar10', help='name of dataset to train with')
    parser.add_argument('--pretrained_path', type=str, default=None, help='resume training of an earlier model if applicable')
    parser.add_argument('--override_hyperparameters', type=bool, default=False, help='train an old model with new hyperparameters')

    # Training Hyperparameters
    parser.add_argument('--data_batch_size', type=int, default=32, help='batch size of samples from real data')
    parser.add_argument('--noise_batch_size', type=int, default=64, help='batch size of samples of random noise')
    parser.add_argument('--dis_iters', type=int, default=5, help='number of times to train discriminator per generator batch')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--subsample', type=float, default=None, help='rate at which to subsample the dataset')
    
    # Evaluation Hyperparameters
    parser.add_argument('--n_fid_imgs', type=int, default=10000, help='number of images to use for FID, should be >= 10000, must be > 2048')
    parser.add_argument('--n_is_imgs', type=int, default=5000, help='number of images to use for evaluating inception score')
    args = parser.parse_args()
    if args.pretrained_path is None and args.override_hyperparameters:
        parser.error('--override_hyperparameters can only be set when loading a previous model with --pretrained-path')

    model_name = strftime("%a, %d %b %Y %H:%M:%S +0000/", gmtime())
    results_path = os.path.join(args.sn_gan_data_path, model_name)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    global logging
    logging.basicConfig(filename=os.path.join(results_path, 'training.log'), level=logging.DEBUG)

    config = {
        'results_path': results_path,
        **vars(args)
    }

    logging.info("Building models")

    if args.pretrained_path is None:
        d = Cifar10Discriminator()
        g = Cifar10Generator()
        d_optim = torch.optim.Adam(d.parameters(), lr=.0002)
        g_optim = torch.optim.Adam(g.parameters(), lr=.0002)
        trainingwrapper = TrainingWrapper(d, g, d_optim, g_optim, config)
    else:
        trainingwrapper = TrainingWrapper.load(args.pretrained_path)
        trainingwrapper.config['epochs'] = args.epochs
        trainingwrapper.config['results_path'] = results_path
        if args.override_hyperparameters:
            trainingwrapper.config = config

    logging.info("Build training wrapper")

    update(trainingwrapper)
        
if __name__ == '__main__':
    main()
