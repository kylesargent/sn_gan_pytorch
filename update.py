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

def sample_c(batch_size, n_classes):
    if n_classes == 0:
        return None
    else:
        return torch.randint(low=0, high=n_classes, size=(batch_size,))

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
    conditional = config['conditional']

    n_is_imgs = config['n_is_imgs']
    n_fid_imgs = config['n_fid_imgs']
    eval_interval = config['eval_interval']
    dataset = config['dataset']
    sn_gan_data_path = config['sn_gan_data_path']
    results_path = config['results_path']

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
    n_classes = dataset['n_classes'] if conditional else 0
    train_iter = dataset['train_iter']

    logging.info("Starting training\n")
    print("Training")

    gen_losses = []
    dis_losses = []

    for iters in tqdm(range(max_iters)):
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
            if conditional:
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

        if (iters + 1) % eval_interval == 0:
            checkpoint = (iters + 1) // eval_interval
            checkpoint_path = os.path.join(results_path, 'checkpoint_{}'.format(checkpoint))

            eval_imgs_path = os.path.join(checkpoint_path, 'eval_imgs/')
            
            os.makedirs(os.path.dirname(eval_imgs_path), exist_ok=True)

            model_save_path = os.path.join(checkpoint_path, 'wrapper.pt')
            trainingwrapper.save(model_save_path)

            g.eval()
                    
            n_imgs = n_fid_imgs
            images = []
            labels = []
            eval_batch_size = 128
            for _ in range(math.ceil(n_fid_imgs / float(eval_batch_size))):
                with torch.no_grad():
                    z = sample_z(eval_batch_size).to(device)
                    c = sample_c(eval_batch_size, n_classes).to(device)
                    images += [g(z, c).cpu()]
                    labels += [c.cpu()]

            images = torch.cat(images)
            images = (images + 1) / 2
            transform = ToPILImage()

            labels = torch.cat(labels)

            for i, (label, image) in enumerate(zip(labels, images)):
                im = transform(image)
                im.save(os.path.join(eval_imgs_path, 'class_{}_image_{}.jpg'.format(label, i)))

            # evaluation - losses
            logging.info("Mean generator loss: {}\n".format(np.mean(gen_losses)))
            logging.info("Mean discriminator loss: {}\n".format(np.mean(dis_losses)))
            gen_losses = []
            dis_losses = []

            # evaluation - is
            images = images.transpose(1,2)
            images = images.transpose(2,3) 
            images = images.numpy() * 256

            inception_score = get_inception_score(list(images))[0]
            logging.info("Inception Score: {}\n".format(inception_score))

            # evaluation - fid
            fid = calculate_fid_given_paths((eval_imgs_path, fid_stats_path), sn_gan_data_path)
            logging.info("FID: {}\n".format(fid))

            g.train()
