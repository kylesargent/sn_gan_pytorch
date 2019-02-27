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

from cifar10_models import sample_z, sample_c

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

def checksum(model):
    return sum(torch.sum(parameter) for parameter in model.parameters())

def train(trainingwrapper, dataset):
    d = trainingwrapper.d.train()
    g = trainingwrapper.g.train()
    d_optim = trainingwrapper.d_optim
    g_optim = trainingwrapper.g_optim
    d_scheduler = trainingwrapper.d_scheduler
    g_scheduler = trainingwrapper.g_scheduler
    config = trainingwrapper.config

    data_batch_size = config['data_batch_size']
    noise_batch_size = config['noise_batch_size']
    dis_iters = config['dis_iters']
    max_iters = config['max_iters']
    subsample = config['subsample']
    conditional = config['conditional']

    sn_gan_data_path = config['sn_gan_data_path']
    results_path = config['results_path']

    global logging
    logging.basicConfig(filename=os.path.join(results_path, 'training.log'), level=logging.DEBUG)

    n_classes = dataset['n_classes'] if conditional else 0
    train_iter = dataset['train_iter']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info("Using device {}\n".format(str(device)))
    d.to(device)
    g.to(device)

    logging.info("Starting training\n")
    print("Training")

    gen_losses = dis_losses = []

    for iters in tqdm(range(max_iters)):
        d_scheduler.step()
        g_scheduler.step()

        for i in range(dis_iters):
            # train generator
            if i == 0:
                for p in d.parameters():
                    p.requires_grad = False

                g_optim.zero_grad()
                z = sample_z(noise_batch_size).to(device)
                y_fake = sample_c(noise_batch_size, n_classes).to(device)

                x_fake = g(z, y_fake)
                dis_fake = d(x_fake, y_fake)
                gen_loss = get_gen_loss_hinge(dis_fake)
                gen_loss.backward()
                g_optim.step()

                for p in d.parameters():
                    p.requires_grad = True

            # train discriminator
            x_real, y_real = next(train_iter)
            x_real = x_real.to(device)
            y_real = y_real.to(device)
            if not conditional:
                y_real = None

            d_optim.zero_grad()
            dis_real = d(x_real, y_real)

            z = sample_z(data_batch_size).to(device)
            y_fake = sample_c(data_batch_size, n_classes).to(device)
            x_fake = g(z, y_fake).detach()
            dis_fake = d(x_fake, y_fake)

            dis_loss = get_dis_loss_hinge(dis_fake, dis_real)
            dis_loss.backward()
            d_optim.step()

        gen_losses += [gen_loss.cpu().data.numpy()]
        dis_losses += [dis_loss.cpu().data.numpy()]

        if (iters + 1) % 100 == 0:
            logging.info("Mean generator loss: {}", np.mean(gen_losses))
            logging.info("Mean discriminator loss: {}", np.mean(dis_losses))
            gen_losses = dis_losses = []
