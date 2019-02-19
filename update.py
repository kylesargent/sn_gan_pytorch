import os
from tqdm import tqdm
import logging

from PIL import Image
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.autograd import Variable

from datasets import get_dataset_struct
from TTUR.fid import calculate_fid_given_paths
from inception_score import get_inception_score


def gen_loss(dis_fake):
    return F.softplus(-dis_fake).mean(0)
    

def dis_loss(dis_fake, dis_real):
    L1 = F.softplus(dis_fake).mean(0)
    L2 = F.softplus(-dis_real).mean(0)    
    return L1 + L2


def sample_z(batch_size):
    n = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    return n.sample((batch_size, 128)).squeeze(2)


def update(trainingwrapper):
    d = trainingwrapper.d
    g = trainingwrapper.g
    d_optim = trainingwrapper.d_optim
    g_optim = trainingwrapper.g_optim
    config = trainingwrapper.config

    dataset = config['dataset']
    sn_gan_data_path = config['sn_gan_data_path']
    results_path = config['results_path']
    eval_imgs_path = config['eval_imgs_path']

    data_batch_size = config['data_batch_size']
    noise_batch_size = config['noise_batch_size']
    dis_iters = config['dis_iters']
    epochs = config['epochs']
    subsample = config['subsample']
    n_is_imgs = config['n_is_imgs']
    n_fid_imgs = config['n_fid_imgs']

    global logging
    logging.basicConfig(filename=os.path.join(results_path, 'training.log'), level=logging.DEBUG)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info("Using device {}".format(str(device)))

    d.to(device)
    g.to(device)

    dataset = get_dataset_struct(dataset, sn_gan_data_path, data_batch_size, subsample)
    train_iter = dataset['train_iter']
    fid_stats_dir = dataset['fid_stats_dir']

    logging.info("Starting training")
    for epoch in range(epochs + 1):
        # training
        
        if epoch != 0:
            # baseline evaluation of a random model

            logging.info("Beginning training epoch {}".format(epoch))

            for batch, _labels in tqdm(train_iter):
                z = Variable(sample_z(noise_batch_size).to(device))
                x_real = Variable(batch.to(device))
                
                for k in range(dis_iters):
                    # train discriminator
                    x_fake = g(z)
                    dis_fake = d(x_fake.detach())
                    dis_real = d(x_real)

                    loss = dis_loss(dis_fake, dis_real)
                    loss.backward()
                    d_optim.step()

                    # train generator
                    if k==0:
                        g_optim.zero_grad()
                        dis_fake = d(x_fake)

                        loss = gen_loss(dis_fake) 
                        loss.backward()
                        g_optim.step()

            checkpoint_path = os.path.join(results_path, 'checkpoints/checkpoint_{}'.format(epoch))
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            trainingwrapper.save(checkpoint_path)

        # evaluation - is
        n_imgs = n_fid_imgs if epoch == epochs - 1 else n_is_imgs
        images = []
        eval_batch_size = 10
        for _ in range(math.ceil(n_imgs / float(eval_batch_size))):
            with torch.no_grad():
                z = sample_z(eval_batch_size).to(device)
                images += [g(z).cpu()]

        images = torch.cat(images)
        images = images.transpose(1, 3)
        images = (images + 1) * 128
        images = images.numpy()

        inception_score = get_inception_score(list(images))[0]
        logging.info("\nInception Score at epoch {}: {}".format(epoch, inception_score))

        images_dir = os.path.join(eval_imgs_path, 'epoch_{}/imgs/'.format(epoch))
        os.makedirs(os.path.dirname(images_dir), exist_ok=True)

        for i, image in enumerate(images):
            im = Image.fromarray(image, 'RGB')
            im.save(os.path.join(images_dir, '{}.jpg'.format(i)))

        # evaluation - fid
        fid = calculate_fid_given_paths((images_dir, fid_stats_dir), sn_gan_data_path)
        logging.info("\nFID at epoch {}: {}".format(epoch, fid))
