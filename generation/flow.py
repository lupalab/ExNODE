import os
import json
import argparse
import time
import numpy as np
import logging
from pprint import pformat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns; sns.set()

import torch
import torch.nn as nn
import torch.optim as optim

from point_cloud import get_loader
from flow_modules import Flow

parser = argparse.ArgumentParser('Flow')
parser.add_argument('--solver', type=str, default='dopri5')
parser.add_argument('--steps', type=int, default=10)
parser.add_argument('--num_blocks', type=int, default=1)
parser.add_argument('--dims', type=str, default='10')
parser.add_argument('--T', type=float, default=0.5)
parser.add_argument('--tol', type=float, default=1e-5)
parser.add_argument('--data', type=str, default='spatial_mnist')
parser.add_argument('--category', type=str, default='airplane')
parser.add_argument('--chn_size', type=int, default=2)
parser.add_argument('--set_size', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=250)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--exp_dir', type=str)
args = parser.parse_args()
args.dims = [int(d) for d in args.dims.split(',')]

args.device = device = torch.device(f'cuda:{args.gpu}')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
np.set_printoptions(precision=4)

os.makedirs(args.exp_dir, exist_ok=True)
logging.basicConfig(filename=args.exp_dir + '/log',
                    filemode='w',
                    level=logging.INFO,
                    format='%(message)s')
logging.info(pformat(vars(args)))

##########################################################


class Model(object):
    def __init__(self, args):
        super(Model, self).__init__()

        self.model = Flow(args).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 100, 0.5)
        
        logging.info(str(self.model))

    def logp(self, x):
        logpx = self.model.logp(x)

        return logpx

    def sample(self, shape):
        x = self.model.sample(shape)

        return x

    def save(self):
        fname = f'{args.exp_dir}/model.pth'
        torch.save(self.model.state_dict(), fname)

    def load(self):
        fname = f'{args.exp_dir}/model.pth'
        self.model.load_state_dict(torch.load(fname))

    def set_data(self, trainloader, testloader):
        self.trainloader = trainloader
        self.testloader = testloader

    def fit(self):
        best_train_ll = -np.inf
        best_test_ll = -np.inf
        for epoch in range(args.epochs):
            train_ll = self.train()
            test_ll = self.test()

            if train_ll > best_train_ll:
                best_train_ll = train_ll
            if test_ll > best_test_ll:
                best_test_ll = test_ll
                self.save()

            msg = f'{time.strftime("%H:%M:%S", time.localtime())} Epoch: {epoch} '
            msg += f'Train: {train_ll:.4f}/{best_train_ll:.4f} '
            msg += f'Test: {test_ll:.4f}/{best_test_ll:.4f}'
            logging.info(msg)

            self.scheduler.step()

    def train(self):
        train_ll = []
        for x in self.trainloader:
            x = x.to(device)
            self.optimizer.zero_grad()
            logpx = self.logp(x)
            loss = -torch.mean(logpx)
            loss.backward()
            self.optimizer.step()
            train_ll.append(logpx.data.cpu().numpy())
        train_ll = np.concatenate(train_ll, axis=0)

        return np.mean(train_ll)

    def test(self):
        test_ll = []
        with torch.no_grad():
            for x in self.testloader:
                x = x.to(device)
                logpx = self.logp(x)
                test_ll.append(logpx.data.cpu().numpy())
        test_ll = np.concatenate(test_ll, axis=0)

        return np.mean(test_ll)

if __name__ == '__main__':
    # data
    trainloader, testloader = get_loader(args)

    # model
    model = Model(args)

    # train
    model.set_data(trainloader, testloader)
    model.fit()

    # sample
    model.load()
    shape = [9, args.set_size, args.chn_size]
    x_sam = model.sample(shape)
    x_sam = x_sam.data.cpu().numpy()

    # plot
    if args.chn_size == 2:
        fig, axs = plt.subplots(3,3, figsize=(5,5))
        for i in range(3):
            for j in range(3):
                ind = j + i * 3
                x = x_sam[ind]
                axs[i,j].scatter(x[:,0], x[:,1])
        plt.savefig(f'{args.exp_dir}/sample.png')
        plt.close('all')
    elif args.chn_size == 3:
        fig = plt.figure()
        for i in range(3):
            for j in range(3):
                ind = j + i * 3
                ax = fig.add_subplot(3, 3, ind+1, projection='3d')
                x = x_sam[ind]
                ax.scatter(x[:,0], x[:,1], x[:,2])
        plt.savefig(f'{args.exp_dir}/sample.png')
        plt.close('all')

