from inception_score import get_inception_score as get_inception_score_tf

from torchvision.transforms import ToPILImage
import torch
import torch.nn as nn
import math
import numpy as np

from cifar10_models import sample_z, sample_c

import logging
import os
from tqdm import tqdm

current_checkpoint = 0

import sys
sys.path.insert(0,'/home/kyle/sn_gan_pytorch')
from sn_gan_pytorch.cifar10_models import sample_z
from sngan_projection_master.source.inception.inception_score import inception_score
from sngan_projection_master.evaluation import load_inception_model


INCEPTION_MODEL_PATH = '/home/kyle/sngan_projection_master/source/inception/inception_score.model'

def save_checkpoint(trainingwrapper):
    config = trainingwrapper.config
    results_path = config['results_path']

    global current_checkpoint
    current_checkpoint += 1
    checkpoint_path = os.path.join(results_path, 'checkpoint_{}'.format(current_checkpoint))
    eval_imgs_path = os.path.join(checkpoint_path, 'eval_imgs/')

    os.makedirs(os.path.dirname(eval_imgs_path), exist_ok=True)
    model_save_path = os.path.join(checkpoint_path, 'wrapper.pt')
    trainingwrapper.save(model_save_path)

def generate_images(gen, n_imgs, batch_size, truncate=False):
    # only works if the model is on one device!!
    device = next(gen.parameters()).device

    with torch.no_grad():
        for _ in tqdm(range(math.ceil(n_imgs / float(eval_batch_size)))):
            z = sample_z(eval_batch_size, truncate=truncate).to(device)
            images += [gen(z).cpu()]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

    images = torch.cat(images).numpy()
    images = np.asarray(np.clip(images * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
    return images.astype("f")

def calc_inception_chainer(images, splits):
    assert(images.shape[1] == 3)
    assert(images.shape[2] == 32)
    assert(images.shape[3] == 32)
    model = load_inception_model(INCEPTION_MODEL_PATH)
    mean, std = inception_score(model, images, splits=splits)
    return mean, std

def calc_inception_tf(gen):
    images = generate_images(gen)

    # evaluation - is
    images = images.transpose(1,2)
    images = images.transpose(2,3) 
    images = images.numpy() * 255.



    g.to('cpu')
    trainingwrapper.d.to('cpu')
    inception_score_mean, inception_score_variance = get_inception_score(list(images))
    g.to(device)
    trainingwrapper.d.to(device)

    print(inception_score_mean, inception_score_variance)
    logging.info("Inception Score: {}+/-{}".format(inception_score_mean, inception_score_variance))
    g.train()
