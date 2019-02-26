import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
from tqdm import tqdm

from PIL import Image
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.autograd import Variable
from torchvision.transforms import ToPILImage

from datasets import get_dataset_struct
from TTUR.fid import calculate_fid_given_paths
from inception_score import get_inception_score



def get_gen_loss(dis_fake):
    return F.softplus(-dis_fake).mean(0)

def get_gen_loss_hinge(dis_fake):
    return -dis_fake.mean(0)


def get_dis_loss(dis_fake, dis_real):
    L1 = F.softplus(dis_fake).mean(0)
    L2 = F.softplus(-dis_real).mean(0)    
    return L1 + L2

def get_dis_loss_hinge(dis_fake, dis_real):
    L1 = F.relu(1 + dis_fake).mean(0)
    L2 = F.relu(1 - dis_real).mean(0)
    return L1 + L2


def sample_z(batch_size):
    n = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    return n.sample((batch_size, 128)).squeeze(2)

def checksum(model):
    return sum(torch.sum(parameter) for parameter in model.parameters())

def update(trainingwrapper):
    d = trainingwrapper.d.train()
    g = trainingwrapper.g.train()
    d_optim = trainingwrapper.d_optim
    g_optim = trainingwrapper.g_optim
    config = trainingwrapper.config

    data_batch_size = config['data_batch_size']
    noise_batch_size = config['noise_batch_size']
    dis_iters = config['dis_iters']
    max_iters = config['max_iters']
    subsample = config['subsample']

    n_is_imgs = config['n_is_imgs']
    n_fid_imgs = config['n_fid_imgs']
    dataset = config['dataset']
    sn_gan_data_path = config['sn_gan_data_path']
    results_path = config['results_path']
    eval_imgs_path = os.path.join(results_path, 'eval_imgs/')

    global logging
    logging.basicConfig(filename=os.path.join(results_path, 'training.log'), level=logging.DEBUG)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info("Using device {}\n".format(str(device)))

    if torch.cuda.device_count() > 1:
        d = nn.DataParallel(d)
        g = nn.DataParallel(g)

    d.to(device)
    g.to(device)

    dataset = get_dataset_struct(dataset, sn_gan_data_path, data_batch_size, subsample)

    fid_stats_path = dataset['fid_stats_path']

    train_iter = dataset['train_iter']
    logging.info("Starting training\n")
    
    for iters in range(max_iters):
        for i in range(dis_iters):
            # train generator
            if i == 0:
                for p in d.parameters():
                    p.requires_grad = False

                g_optim.zero_grad()
                z = sample_z(noise_batch_size).to(device)
                x_fake = g(z)
                dis_fake = d(x_fake)
                gen_loss = get_gen_loss_hinge(dis_fake)
                gen_loss.backward()
                g_optim.step()

                for p in d.parameters():
                    p.requires_grad = True

            # train discriminator
            batch, _labels = next(train_iter)

            d_optim.zero_grad()
            x_real = batch.to(device)
            dis_real = d(x_real)
            z = sample_z(data_batch_size).to(device)
            x_fake = g(z).detach()

            dis_fake = d(x_fake)
            dis_loss = get_dis_loss_hinge(dis_fake, dis_real)
            dis_loss.backward()
            d_optim.step()

        # gen_losses += [gen_loss.cpu().data.numpy()]
        # dis_losses += [dis_loss.cpu().data.numpy()]

        # logging.info("Mean generator loss: {}\n".format(np.mean(gen_losses)))
        # logging.info("Mean discriminator loss: {}\n".format(np.mean(dis_losses)))

        # checkpoint_path = os.path.join(results_path, 'checkpoints/checkpoint_{}'.format(epoch))
        # os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        # trainingwrapper.save(checkpoint_path)

    g.eval()


            
    n_imgs = n_fid_imgs
    images = []
    eval_batch_size = 128
    for _ in range(math.ceil(n_fid_imgs / float(eval_batch_size))):
        with torch.no_grad():
            z = sample_z(eval_batch_size).to(device)
            images += [g(z).cpu()]

    images = torch.cat(images)
    images = (images + 1) / 2

    images_path = os.path.join(eval_imgs_path, 'epoch_{}/'.format(epoch))
    os.makedirs(os.path.dirname(images_path), exist_ok=True)
    transform = ToPILImage()

    for i, image in enumerate(images):
        im = transform(image)
        im.save(os.path.join(images_path, '{}.jpg'.format(i)))

    # evaluation - is
    inception_score = get_inception_score(list(images.numpy() * 256))[0]
    logging.info("Inception Score at epoch {}: {}\n".format(epoch, inception_score))

    # evaluation - fid
    fid = calculate_fid_given_paths((images_path, fid_stats_path), sn_gan_data_path)
    logging.info("FID at epoch {}: {}\n".format(epoch, fid))
