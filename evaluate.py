from TTUR.fid import calculate_fid_given_paths
from inception_score import get_inception_score

from torchvision.transforms import ToPILImage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import math
import numpy as np

from PIL import Image
from cifar10_models import sample_z, sample_c

import logging
import os

from tqdm import tqdm

current_checkpoint = 0

def evaluate(trainingwrapper, dataset):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config = trainingwrapper.config
    sn_gan_data_path = config['sn_gan_data_path']
    results_path = config['results_path']
    conditional = config['conditional']

    global logging
    logging.basicConfig(filename=os.path.join(results_path, 'training.log'), level=logging.DEBUG)

    n_classes = dataset['n_classes'] if conditional else 0
    fid_stats_path = dataset['fid_stats_path']

    g = trainingwrapper.g

    n_is_imgs = config['n_is_imgs']
    n_fid_imgs = config['n_fid_imgs']
    eval_batch_size = config['eval_batch_size']
    truncate = config['truncate']
    logging.info('Computing examples with truncation={}'.format(truncate))

    ###
    global current_checkpoint
    current_checkpoint += 1
    checkpoint_path = os.path.join(results_path, 'checkpoint_{}'.format(current_checkpoint))
    eval_imgs_path = os.path.join(checkpoint_path, 'eval_imgs/')
    
    os.makedirs(os.path.dirname(eval_imgs_path), exist_ok=True)

    model_save_path = os.path.join(checkpoint_path, 'wrapper.pt')
    trainingwrapper.save(model_save_path)

    g.to(device)
    g.eval()
            
    n_imgs = n_fid_imgs
    images = []
    labels = []
    with torch.no_grad():
        for _ in tqdm(range(math.ceil(n_fid_imgs / float(eval_batch_size)))):
            z = sample_z(eval_batch_size, truncate=truncate).to(device)
            c = sample_c(eval_batch_size, n_classes)
            if c is not None:
                c = c.to(device)
            images += [g(z, c).cpu()]
            if c is not None:
                c = c.cpu()
            else:
                c = 0
            labels += [c]

    images = torch.cat(images)
    images = (images + 1) / 2
    transform = ToPILImage()

    labels = torch.cat(labels)

    class_counts = np.zeros(n_classes).astype(int)
    for label, image in zip(labels, images):
        im = transform(image)
        im.save(os.path.join(eval_imgs_path, 'class_{}_image_{}.jpg'.format(label, class_counts[label])))
        class_counts[label] += 1

    # evaluation - is
    images = images.transpose(1,2)
    images = images.transpose(2,3) 
    images = images.numpy() * 256

    inception_score_mean, inception_score_variance = get_inception_score(list(images))
    logging.info("Inception Score: {}+/-{}".format(inception_score_mean, inception_score_variance))

    # evaluation - fid
    fid = calculate_fid_given_paths((eval_imgs_path, fid_stats_path), sn_gan_data_path)
    logging.info("FID: {}\n".format(fid))

    g.train()