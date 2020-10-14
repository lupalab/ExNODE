import os
import math
import torch
import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from torchdiffeq import odeint as odeint_normal
from torchdiffeq import odeint_adjoint as odeint


def set_optimizer(parameters, args):
    # select optimizer
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(parameters, lr=args.lr, betas=(args.beta1, args.beta2))
    elif args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(parameters, lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(parameters, lr=args.lr)
    else:
        raise NotImplementedError(f'Optimizer {args.optimizer} is not understood.')
    
    return optimizer


class DeepsetBlock(nn.Module):
    def __init__(self, i_dim, h_dims):
        super(DeepsetBlock, self).__init__()

        self.encode = FCnet(i_dim, h_dims)
        self.max = Max(1)
        self.fc = nn.Linear(h_dims[-1], i_dim)

    def forward(self, t, x):
        x = self.encode(x)
        x = x - self.max(x)
        x = self.fc(x)
        return x


class Deepset(nn.Module):
    def __init__(self, args):
        super(Deepset, self).__init__()

        class dsetblock(nn.Module):
            def __init__(self, i_dim, h_dims):
                super(dsetblock, self).__init__()

                self.encode = FCnet(i_dim, h_dims)
                self.max = Max(1)
                self.fc = nn.Linear(h_dims[-1], i_dim)

            def forward(self, x):
                x = self.encode(x)
                x = x - self.max(x)
                x = self.fc(x)
                return x

        self.feature_extractor = FCnet(args.pts_dim, args.dims)
        self.deepset = nn.Sequential(
            *[dsetblock(args.dims[-1], args.set_hdims) for _ in range(args.num_blocks)])
        self.fc = FCnet(args.dims[-1], args.fc_dims)
        self.logit = nn.Linear(args.fc_dims[-1], 40)
        self.max = Max(1)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.deepset(x)
        x = self.max(x)
        x = x.view(x.shape[0], -1)
        x = self.logit(self.fc(x))
        return x


def FCnet(in_dim, h_dims):
    net = []
    for h in h_dims[:-1]:
        net.append(nn.Linear(in_dim, h))
        net.append(nn.Tanh())
        in_dim = h
    net.append(nn.Linear(in_dim, h_dims[-1]))

    return nn.Sequential(*net)


class SmallTransformer(nn.Module):
    
    def __init__(self, i_dim, h_dims, num_head=4):
        super(SmallTransformer, self).__init__()
        self.dim = h_dims[-1]
        self.num_head = num_head
        self.K = FCnet(i_dim, h_dims)
        self.Q = FCnet(i_dim, h_dims)
        self.V = FCnet(i_dim, h_dims)
        self.M = nn.Linear(h_dims[-1], i_dim)
    
    def encode(self, x):
        batch_size = x.shape[0]
        k = self.K(x)
        q = self.Q(x)
        v = self.V(x)
        head_size = self.dim // self.num_head
        k = torch.cat(k.split(head_size, dim=2), dim=0)
        q = torch.cat(q.split(head_size, dim=2), dim=0)
        v = torch.cat(v.split(head_size, dim=2), dim=0)
        A = q.bmm(k.transpose(1,2)) / math.sqrt(head_size)
        A = torch.softmax(A, dim=2)
        r = torch.cat((q + A.bmm(v)).split(batch_size, dim=0), dim=2)
        r = self.M(torch.tanh(r))
        return r
    
    def forward(self, t, x):
        x = self.encode(x)
        return x


class ODEBlock(nn.Module):

    def __init__(self, odefunc, T, steps, rtol, atol, solver):
        super(ODEBlock, self).__init__()
        
        self.odefunc = odefunc
        self.integration_time = torch.linspace(0.0, T, steps).float()

        self.rtol = rtol
        self.atol = atol
        self.solver = solver

    def forward(self, x):
        self.integration_time = self.integration_time.to(x)
        if self.solver != 'dopri5':
            out = odeint_normal(self.odefunc, x, self.integration_time, self.rtol, self.atol, self.solver)
        else:
            out = odeint(self.odefunc, x, self.integration_time, self.rtol, self.atol, self.solver)
        return out[-1]
    
    def logits(self, x):
        return self.forward(x)
    
    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nef(self, value):
        self.odefunc.nfe = value


class Max(nn.Module):
    def __init__(self, dim):
        super(Max, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return torch.max(x, self.dim, keepdim=True)[0]


class Flatten(nn.Module):
    def __init__(self, ):
        super(Flatten, self).__init__()
    
    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class Transpose(nn.Module):
    def __init__(self, i_dim, j_dim):
        super(Transpose, self).__init__()

        self.i_dim = i_dim
        self.j_dim = j_dim
    
    def forward(self, x):
        return x.transpose(self.i_dim, self.j_dim)


class ODEModel(nn.Module):
    def __init__(self, args):
        super(ODEModel, self).__init__()

        self.args = args

        feature_extractor = nn.ModuleList()
        
        layer_dims = args.dims.copy()
        batch_norm = args.batch_norm
        if not isinstance(layer_dims, list):
            layer_dims = list(layer_dims)
        layer_dims.insert(0, args.pts_dim)

        for idx in range(len(layer_dims) - 1):
            feature_extractor.append(nn.Conv1d(layer_dims[idx], layer_dims[idx+1], 1))
            if batch_norm:
                feature_extractor.append(nn.BatchNorm1d(layer_dims[idx+1]))
            feature_extractor.append(nn.Tanh())
        feature_extractor.append(Transpose(1, 2))
        
        if args.sub_model == 'odedset':
            feature_layers = [ODEBlock(DeepsetBlock(args.dims[-1], args.set_hdims), args.T_end, args.steps, args.tol, args.tol, args.solver)
                for _ in range(args.num_blocks)]
        elif args.sub_model == 'odetrans':
            feature_layers = [ODEBlock(SmallTransformer(args.dims[-1], args.set_hdims), args.T_end, args.steps, args.tol, args.tol, args.solver)
                for _ in range(args.num_blocks)]
        else:
            raise NotImplementedError('the input diffeq model is not understood')

        fc_dims = args.fc_dims.copy()
        if not isinstance(fc_dims, list):
            fc_dims = list(fc_dims)
        fc_dims.insert(0, args.dims[-1])

        fc_layers = nn.ModuleList()
        fc_layers.append(Max(1)); fc_layers.append(Flatten())

        for idx in range(len(fc_dims) - 1):
            fc_layers.append(nn.Linear(fc_dims[idx], fc_dims[idx+1]))
            if batch_norm:
                fc_layers.append(nn.BatchNorm1d(fc_dims[idx+1]))
            fc_layers.append(nn.Tanh())
        
        fc_layers.append(nn.Linear(fc_dims[-1], 40))

        self.model = nn.Sequential(*feature_extractor, *feature_layers, *fc_layers)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.model(x)
        return x


class ClassificationEngine(object):

    def __init__(self, args, logger):
        super(ClassificationEngine, self).__init__()

        self.args = args
        self.logger = logger
        self.device = args.device
        
        if args.sub_model == 'deepset':
            self.model = Deepset(args).to(self.device)
        else:
            self.model = ODEModel(args).to(self.device)

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = set_optimizer(self.model.parameters(), args)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 20, 0.5)

    def logits(self, x):
        logits_x = self.model(x)

        return logits_x

    def save(self):
        fname = os.path.join(self.args.save, 'model.pth')
        torch.save(self.model.state_dict(), fname)

    def load(self):
        fname = os.path.join(self.args.save, 'model.pth')
        self.model.load_state_dict(torch.load(fname))

    def set_data(self, trainloader, testloader):
        self.trainloader = trainloader
        self.testloader = testloader
    
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def fit(self):
        best_train_acc = -np.inf
        best_test_acc = -np.inf
        metric_log = {
            'Epoch': np.linspace(1, self.args.epochs, self.args.epochs),
            'train_acc': [],
            'test_acc': []
        }
        for epoch in range(self.args.epochs):
            train_acc = self.train()
            test_acc = self.test()

            if train_acc > best_train_acc:
                best_train_acc = train_acc
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                self.save()

            msg = f'{time.strftime("%H:%M:%S", time.localtime())} Epoch: {epoch} '
            msg += f'Train: {train_acc:.4f}/{best_train_acc:.4f} '
            msg += f'Test: {test_acc:.4f}/{best_test_acc:.4f} '
            self.logger.info(msg)

            self.scheduler.step()

            metric_log['train_acc'].append(train_acc)
            metric_log['test_acc'].append(test_acc)
        
        return metric_log
    
    def one_hot(self, x, K):
        return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)
    
    def train(self):
        self.model.train()
        train_acc = []
        for idx, (x, y) in enumerate(tqdm(self.trainloader)):
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.logits(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
                
            y = self.one_hot(np.array(y.detach().cpu().numpy()), 40)
            target_class = np.argmax(y, axis=1)
            predicted_class = np.argmax(logits.detach().cpu().numpy(), axis=1)
            acc = np.sum(predicted_class == target_class) / float(x.shape[0])
            train_acc.append(acc)
        train_acc = np.array(train_acc)

        return np.mean(train_acc)

    def test(self):
        self.model.eval()
        test_acc = []
        with torch.no_grad():
            for idx, (x, y) in enumerate(tqdm(self.testloader)):
                x = x.to(self.device)
                logits = self.logits(x)
                
                y = self.one_hot(np.array(y.numpy()), 40)
                target_class = np.argmax(y, axis=1)
                predicted_class = np.argmax(logits.detach().cpu().numpy(), axis=1)
                acc = np.sum(predicted_class == target_class) / float(x.shape[0])
                test_acc.append(acc)
        test_acc = np.array(test_acc)

        return np.mean(test_acc)