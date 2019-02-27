import argparse
import os
from os.path import expanduser
import logging
from time import gmtime, strftime

from cifar10_models import Cifar10Generator, Cifar10Discriminator
from trainingwrapper import TrainingWrapper
from train import train
from evaluate import evaluate

import torch
from datasets import get_dataset_struct

DEFAULT_SN_GAN_DATA_PATH = os.path.expanduser('~/sn_gan_pytorch_data')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sn_gan_data_path', type=str, default=DEFAULT_SN_GAN_DATA_PATH, help='where to put model data and downloads')
    parser.add_argument('--dataset', type=str, default='cifar10', help='name of dataset to train with')
    parser.add_argument('--pretrained_path', type=str, default=None, help='resume training of an earlier model if applicable')
    parser.add_argument('--override_hyperparameters', type=bool, default=False, help='train the states of an old model with new hyperparameters')

    # Architecture Parameters
    parser.add_argument('--conditional', type=bool, default=True, help='Train a conditional GAN')

    # Training Hyperparameters
    parser.add_argument('--data_batch_size', type=int, default=64, help='batch size of samples from real data')
    parser.add_argument('--noise_batch_size', type=int, default=128, help='batch size of samples of random noise')
    parser.add_argument('--dis_iters', type=int, default=5, help='number of times to train discriminator per generator batch')
    parser.add_argument('--max_iters', type=int, default=50000, help='number of training iterations')
    parser.add_argument('--subsample', type=float, default=None, help='rate at which to subsample the dataset')
    
    # Evaluation Hyperparameters
    parser.add_argument('--n_fid_imgs', type=int, default=10000, help='number of images to use for FID, should be >= 10000, must be > 2048')
    parser.add_argument('--n_is_imgs', type=int, default=5000, help='number of images to use for evaluating inception score')
    parser.add_argument('--eval_batch_size', type=int, default=512, help='generate images for evaluation in batches so as not to overload model')

    parser.add_argument('--eval_interval', type=int, default=5000, help='how often to report training statistics')
    parser.add_argument('--dry_run', action='store_true', help='debug on a small subset of training data, and limit evaluation')
    parser.add_argument('--no_truncation', action='store_false', help='don\'t generate images with truncated noise trick during evaluation')

    args = parser.parse_args()
    if args.pretrained_path is None and args.override_hyperparameters:
        parser.error('--override_hyperparameters can only be set when loading a previous model with --pretrained-path')
    if args.max_iters % args.eval_interval != 0:
        parser.error('--eval_interval must divide --max_iters')
    if args.dry_run:
        args.n_fid_imgs = 100
        args.n_is_imgs = 10

        args.subsample = .001

        args.max_iters = 4
        args.eval_interval = 4

        args.data_batch_size = 4
        args.noise_batch_size = 2
        args.eval_batch_size = args.n_fid_imgs
        if args.pretrained_path is not None:
            args.override_hyperparameters = True

    model_name = strftime("%a%d%b%Y__%H_%M_%S/", gmtime())
    results_path = os.path.join(args.sn_gan_data_path, model_name)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print("USING MULTIPLE GPUS")
        d = nn.DataParallel(d)
        g = nn.DataParallel(g)

        args.max_iters = args.max_iters // num_gpus
        args.eval_interval /= num_gpus

        args.noise_batch_size *= num_gpus
        args.data_batch_size *= num_gpus
        args.eval_batch_size *= num_gpus

    num_workers = max(1, 4 * num_gpus)

    dataset = get_dataset_struct(args.dataset, args.sn_gan_data_path, args.data_batch_size, num_workers, args.subsample)

    config = {
        'results_path': results_path,
        'num_gpus': num_gpus,
        'num_workers': num_workers,
        'n_classes': dataset['n_classes'],
        **vars(args)
    }

    global logging
    logging.basicConfig(filename=os.path.join(results_path, 'training.log'), level=logging.DEBUG)
    logging.info("Building models")

    if args.pretrained_path is None:
        d = Cifar10Discriminator(n_classes=10 if args.conditional else 0)
        g = Cifar10Generator(n_classes=10 if args.conditional else 0)
        d_optim = torch.optim.Adam(d.parameters(), lr=.0002, betas=(0.0, 0.9))
        g_optim = torch.optim.Adam(g.parameters(), lr=.0002, betas=(0.0, 0.9))

        linear_decay = lambda iteration: 1 - (iteration / args.max_iters)
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
