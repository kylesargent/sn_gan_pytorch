import math
import numpy as np
from PIL import Image

import argparse
from tqdm import tqdm
import os
from os.path import expanduser
from time import strftime, gmtime
import logging

import torch
from torch.distributions.normal import Normal
import torch.nn.functional as F
from torch.autograd import Variable

from datasets import get_dataset_struct
from cifar10_models import Cifar10Generator, Cifar10Discriminator
from inception_score import get_inception_score

from TTUR.fid import calculate_fid_given_paths



def gen_loss(dis_fake):
    return F.softplus(-dis_fake).mean(0)
    

def dis_loss(dis_fake, dis_real):
    L1 = F.softplus(dis_fake).mean(0)
    L2 = F.softplus(-dis_real).mean(0)    
    return L1 + L2


def sample_z(batch_size):
    n = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    return n.sample((batch_size, 128)).squeeze(2)


def main():
    sn_gan_data_path = os.path.expanduser('~/sn_gan_pytorch_data')
    model_name = strftime("%a, %d %b %Y %H:%M:%S +0000/", gmtime())
    results_path = os.path.join(sn_gan_data_path, model_name)
    


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', help='name of dataset to train with')
    parser.add_argument('--eval_imgs_path', type=str, default=os.path.join(results_path, 'eval_imgs/'), help='path to evaluation images')
    # parser.add_argument('--gpu', type=int, default=0, help='index of gpu to be used')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--gen_batch_size', type=int, default=64, help='generated samples batch size')
    parser.add_argument('--dis_iters', type=int, default=5, help='number of times to train discriminator per generator batch')
    parser.add_argument('--epochs', type=int, default=2, help='number of training epochs')
    parser.add_argument('--subsample', type=float, default=None, help='rate at which to subsample the dataset')
    
    parser.add_argument('--n_fid_imgs', type=int, default=2048, 
        help='number of images to use for evaluating FID, should be >= 10000 or FID will underreport, and must be > 2048')
    parser.add_argument('--n_is_imgs', type=int, default=512, help='number of images to use for evaluating inception score')

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.eval_imgs_path), exist_ok=True)

    global logging
    logging.basicConfig(filename=os.path.join(results_path, 'training.log'), level=logging.DEBUG)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info("using device: {}".format(device))

    batch_size = args.batch_size
    gen_batch_size = args.gen_batch_size
    dis_iters = args.dis_iters
    epochs = args.epochs
    subsample = args.subsample

    dataset = get_dataset_struct(args.dataset, sn_gan_data_path, batch_size, subsample)
    train_iter = dataset['train_iter']
    logging.info("fetched dataset")    

    if str(device) == 'cuda:0':
        logging.info('Allocated: ', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB\n')
        logging.info('Cached: ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB\n')

    G = Cifar10Generator().to(device)
    D = Cifar10Discriminator().to(device)
    g_optim = torch.optim.Adam(G.parameters(), lr=.0002)
    d_optim = torch.optim.Adam(D.parameters(), lr=.0002)
    logging.info("model and optimizers loaded")

    logging.info("starting training")
    for epoch in range(epochs):
        # training

        if epoch != 0:
            # baseline evaluation of a random model

            logging.info("training epoch {}".format(epoch))

            for batch, _labels in tqdm(train_iter):
                z = Variable(sample_z(gen_batch_size).to(device))
                x_real = Variable(batch.to(device))
                
                for k in range(dis_iters):
                    # train discriminator
                    x_fake = G(z)
                    dis_fake = D(x_fake.detach())
                    dis_real = D(x_real)

                    loss = dis_loss(dis_fake, dis_real)
                    loss.backward()
                    d_optim.step()

                    # train generator
                    if k==0:
                        g_optim.zero_grad()
                        dis_fake = D(x_fake)

                        loss = gen_loss(dis_fake) 
                        loss.backward()
                        g_optim.step()

        # evaluation - is
        n_imgs = args.n_fid_imgs if epoch == epochs - 1 else args.n_is_imgs
        images = []
        eval_batch_size = 128
        for _ in range(math.ceil(n_imgs / float(eval_batch_size))):
            z = Variable(sample_z(eval_batch_size)).to(device)
            images += [G(z)]

        images = torch.cat(images)
        images = images.transpose(1, 3)
        images = (images + 1) * 128
        images = images.data.numpy()

        print("Calculating IS: ")
        inception_score = get_inception_score(list(images))[0]
        logging.info("\nInception Score at epoch {}: {}".format(epoch, inception_score))

        images_dir = os.path.join(args.eval_imgs_path, 'epoch_{}_imgs/'.format(epoch))
        os.makedirs(os.path.dirname(images_dir), exist_ok=True)

        for i, image in enumerate(images):
            im = Image.fromarray(image, 'RGB')
            im.save(os.path.join(images_dir, '{}.jpg'.format(i)))

        # evaluation - fid
        print("Calculating FID: ")
        fid = calculate_fid_given_paths((images_dir, dataset['fid_stats_dir']), sn_gan_data_path)
        logging.info("\nFID at epoch {}: {}".format(epoch, fid))
        
if __name__ == '__main__':
    main()
