import argparse
import os
from os.path import expanduser
import logging
from time import gmtime, strftime

from cifar10_models import Cifar10Generator, SNCifar10Generator, Cifar10Discriminator
from trainingwrapper import TrainingWrapper
from train import train
from evaluate import evaluate

import torch
import torch.nn as nn
from datasets import get_dataset_struct

DEFAULT_SN_GAN_DATA_PATH = os.path.expanduser('~/sn_gan_pytorch_data')

def main():
    parser = argparse.ArgumentParser()

    # Run parameters
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to run the model on')
    parser.add_argument('--sn_gan_data_path', type=str, default=DEFAULT_SN_GAN_DATA_PATH, help='where to put model data and downloads')
    parser.add_argument('--dataset', type=str, default='cifar10', help='name of dataset to train with')
    parser.add_argument('--pretrained_path', type=str, default=None, help='resume training of an earlier model if applicable')
    parser.add_argument('--override_hyperparameters', type=bool, default=False, help='train the states of an old model with new hyperparameters')
    parser.add_argument('--model_name', type=str, default=strftime("%a%d%b%Y__%H_%M_%S/", gmtime()), help='name of the model')

    # Architecture Parameters
    parser.add_argument('--conditional', type=bool, default=False, help='Train a conditional GAN')

    # Training Hyperparameters
    parser.add_argument('--data_batch_size', type=int, default=64, help='batch size of samples from real data')
    parser.add_argument('--noise_batch_size', type=int, default=128, help='batch size of samples of random noise')
    parser.add_argument('--dis_iters', type=int, default=5, help='number of times to train discriminator per generator batch')
    parser.add_argument('--max_iters', type=int, default=50000, help='number of training iterations')
    parser.add_argument('--subsample', type=float, default=None, help='rate at which to subsample the dataset')
    parser.add_argument('--loss_type', type=str, default='hinge', help='type of loss to use for GAN')
    
    # Evaluation Hyperparameters
    parser.add_argument('--n_is_imgs', type=int, default=50000, help='number of images to use for evaluating inception score')
    parser.add_argument('--eval_batch_size', type=int, default=512, help='generate images for evaluation in batches so as not to overload model')

    parser.add_argument('--dry_run', action='store_true', help='debug on a small subset of training data, and limit evaluation')
    parser.add_argument('--truncate', action='store_true', help='generate images with truncated noise trick during evaluation')

    # Extensions
    parser.add_argument('--reparametrize', action='store_true', help='whether to reparametrize spectral norms')
    parser.add_argument('--lam1', type=float, default=.05, help='lipschitz deviation penalty')

    parser.add_argument('--use_gp', action='store_true', help='whether to add a gradient penalty')
    parser.add_argument('--lam2', type=float, default=10., help='gradient penalty scaling hyperparameter')

    parser.add_argument('--sn_generator', action='store_true', help='Whether to use spectral normalization in the generator')
    parser.add_argument('--lam3', type=float, default=.1, help='rank loss penalty scaling factor')


    args = parser.parse_args()
    if args.pretrained_path is None and args.override_hyperparameters:
        parser.error('--override_hyperparameters can only be set when loading a previous model with --pretrained-path')
    if args.dry_run:
        args.n_is_imgs = 10

        args.subsample = .001

        args.max_iters = 4

        args.data_batch_size = 4
        args.noise_batch_size = 2
        args.eval_batch_size = args.n_is_imgs
        if args.pretrained_path is not None:
            args.override_hyperparameters = True

    model_name = args.model_name + '/'
    results_path = os.path.join(args.sn_gan_data_path, model_name)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    global logging
    logging.basicConfig(filename=os.path.join(results_path, 'training.log'), level=logging.DEBUG)
    logging.info("Building models")

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print("Multiple GPUs detected")
        logging.info("Multiple GPUs detected")
        args.max_iters //= num_gpus
        args.noise_batch_size *= num_gpus
        args.data_batch_size *= num_gpus

        args.eval_batch_size *= num_gpus

    dataset = get_dataset_struct(args.dataset, args.sn_gan_data_path, args.data_batch_size, 4, args.subsample)

    config = {
        'results_path': results_path,
        'n_classes': dataset['n_classes'],
        **vars(args)
    }

    if args.pretrained_path is None:
        d = Cifar10Discriminator(n_classes=10 if args.conditional else 0, use_gamma=args.reparametrize)

        if config['sn_generator']:
            g = SNCifar10Generator(n_classes=10 if args.conditional else 0)
        else:
            g = Cifar10Generator(n_classes=10 if args.conditional else 0)
        d_optim = torch.optim.Adam(d.parameters(), lr=.0002, betas=(0.0, 0.9))
        g_optim = torch.optim.Adam(g.parameters(), lr=.0002, betas=(0.0, 0.9))

        linear_decay = lambda iteration: 1. - (iteration / args.max_iters)
        d_scheduler = torch.optim.lr_scheduler.LambdaLR(d_optim, linear_decay, -1)
        g_scheduler = torch.optim.lr_scheduler.LambdaLR(g_optim, linear_decay, -1)

        trainingwrapper = TrainingWrapper(d, g, d_optim, g_optim, d_scheduler, g_scheduler, config)
    else:
        trainingwrapper = TrainingWrapper.load(args.pretrained_path)
        trainingwrapper.config['max_iters'] = args.max_iters
        trainingwrapper.config['results_path'] = results_path
        if args.override_hyperparameters:
            trainingwrapper.config = config

    logging.info("Built models and training wrapper")

    train(trainingwrapper, dataset)
    evaluate(trainingwrapper, dataset)

        
if __name__ == '__main__':
    main()
