import argparse
from tqdm import tqdm
import torch
from torch.distributions.normal import Normal
import torch.nn.functional as F
from datasets import get_dataset_iter
from torch.autograd import Variable
from cifar10_models import Cifar10Generator, Cifar10Discriminator

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', help='name of dataset to train with')
    parser.add_argument('--dataset_path', type=str, default='~/', help='path to dataset')
    # parser.add_argument('--gpu', type=int, default=0, help='index of gpu to be used')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--gen_batch_size', type=int, default=64, help='generated samples batch size')
    parser.add_argument('--dis_iters', type=int, default=5, help='number of times to train discriminator per generator batch')
    parser.add_argument('--epochs', type=int, default=1, help='number of training epochs')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("using device: {}\n".format(device))

    args = parser.parse_args()

    batch_size = args.batch_size
    gen_batch_size = args.gen_batch_size
    dis_iters = args.dis_iters
    epochs = args.epochs

    train_iter = get_dataset_iter(args.dataset, args.dataset_path, batch_size)
    print("fetched dataset\n")

    print('Allocated: ', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB\n')
    print('Cached: ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB\n')

    G = Cifar10Generator().to(device)
    D = Cifar10Discriminator().to(device)
    g_optim = torch.optim.Adam(G.parameters(), lr=.0002)
    d_optim = torch.optim.Adam(D.parameters(), lr=.0002)
    print("model and optimizers loaded\n")

    print("starting training\n")
    for epoch in range(epochs):
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

if __name__ == '__main__':
    main()