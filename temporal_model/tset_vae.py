import os
import logging
import json
import copy
import argparse
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import imageio
import scipy
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from time_series import get_loader as get_ts_loader
from dataset import get_loader
from tqdm import tqdm
from math import log, pi
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint as odeint_normal


parser = argparse.ArgumentParser(description="Set Series Modeling")

parser.add_argument('-data', type=str, default='spmnist', help='data to use')
parser.add_argument('-rtol', type=float, default=1e-3, help="integral tolerance")
parser.add_argument('-atol', type=float, default=1e-4, help="integral tolerance")

parser.add_argument('-batch_size', type=int, default=128, help='the batch size to use')
parser.add_argument('-solver', type=str, choices=['euler', 'dopri5', 'rk4'], default='dopri5', help='ode numerical solver')
parser.add_argument('-input_dim', type=int, default=2, help='the dimension of data')
parser.add_argument('-z_dim', type=int, default=128, help='the latent dimension to use')
parser.add_argument('-en_dims', type=str, default='128,128,256,512')
parser.add_argument('-f_dims', type=str, default='512,512,512')

parser.add_argument('-N', type=int, default=50, help='sub-sampling number')
parser.add_argument('-rnn_dim', type=int, default=512, help='the rnn dim to use')
parser.add_argument('-T', type=int, default=5, help='the time steps to use')
parser.add_argument('-T_end', type=float, default=1.0, help='integral end time')
parser.add_argument('-train_T', type=eval, default=True, choices=[True, False], help='whether train T in flow')
parser.add_argument('-num_blocks', type=int, default=1, help='# of blocks to use')
parser.add_argument('-use_adjoint', type=bool, default=True, help='whether to use adjoint method')

parser.add_argument('-epochs', type=int, default=100, help='epochs to train the model')
parser.add_argument('-exp_dir', type=str, default='./exp/spmnist7', help='the path to exp dir')
parser.add_argument('-gpu', type=int, default=0, help='the gpu to use')
parser.add_argument('-seed', type=int, default=123, help='the seed to control randomness')

args = parser.parse_args()

args.en_dims = tuple(map(int, args.en_dims.split(',')))
args.f_dims = tuple(map(int, args.f_dims.split(',')))

device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else 'cpu'
np.random.seed(args.seed)
torch.manual_seed(args.seed)
np.set_printoptions(precision=4)

os.makedirs(args.exp_dir, exist_ok=True)
logging.basicConfig(filename=args.exp_dir + '/log',
                    filemode='w',
                    level=logging.INFO,
                    format='%(message)s')
logging.info(json.dumps(vars(args),indent=4))


def divergence_fn(f, y, e=None):
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx.mul(e)

    cnt = 0
    while not e_dzdx_e.requires_grad and cnt < 10:
        e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
        e_dzdx_e = e_dzdx * e
        cnt += 1

    approx_tr_dzdx = e_dzdx_e.sum(dim=-1)
    assert approx_tr_dzdx.requires_grad, \
        "(failed to add node to graph) f=%s %s, y(rgrad)=%s, e_dzdx:%s, e:%s, e_dzdx_e:%s cnt:%s" \
        % (
        f.size(), f.requires_grad, y.requires_grad, e_dzdx.requires_grad, e.requires_grad, e_dzdx_e.requires_grad, cnt)
    return approx_tr_dzdx


class SetEncoder(nn.Module):
    def __init__(self, input_dim, en_dims):
        super(SetEncoder, self).__init__()

        self.input_dim = input_dim
        dims = en_dims

        layers = []
        layers.append(nn.Conv1d(self.input_dim, dims[0], 1))
        layers.append(nn.BatchNorm1d(dims[0]))
        layers.append(nn.ReLU(True))

        for idx in range(len(dims) - 1):
            layers.append(nn.Conv1d(dims[idx], dims[idx+1], 1))
            layers.append(nn.BatchNorm1d(dims[idx+1]))
            if idx < len(dims) - 2: 
                layers.append(nn.ReLU(True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        B, T, K, D = x.size()
        x = x.permute(0, 1, 3, 2).reshape(B*T, D, K) # x: B*T x D x K
        x = self.layers(x)
        x = torch.max(x, dim=2)[0]
        x = x.view(B, T, -1)
        return x # B x T x H


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1 + dim_c, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1 + dim_c, dim_out)

    def forward(self, context, x):
        gate = torch.sigmoid(self._hyper_gate(context))
        bias = self._hyper_bias(context)
        if x.dim() == 3:
            gate = gate.unsqueeze(1)
            bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret


class ODEnet(nn.Module):
    def __init__(self, hidden_dims, input_shape, context_dim):
        super(ODEnet, self).__init__()
        
        layers = []
        activation_fns = []
        hidden_shape = input_shape
        
        for dim_out in (hidden_dims + (input_shape[0],)):
            layer_kwargs = {}
            layer = ConcatSquashLinear(hidden_shape[0], dim_out, context_dim, **layer_kwargs)
            layers.append(layer)
            activation_fns.append(nn.Tanh())

            hidden_shape = list(copy.copy(hidden_shape))
            hidden_shape[0] = dim_out
        
        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])

    def forward(self, context, y):
        dx = y
        for l, layer in enumerate(self.layers):
            dx = layer(context, dx)
            
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        
        return dx


class ODEfunc_Flow(nn.Module):
    def __init__(self, diffeq):
        super(ODEfunc_Flow, self).__init__()
        self.diffeq = diffeq
        self.divergence = divergence_fn
        self.register_buffer("_num_evals", torch.tensor(0.))

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)
    
    def forward(self, t, states):
        assert len(states) >= 2, "the length of states should be >= 2"
        y = states[0]
        t = torch.ones(y.size(0), 1).to(y) * t.clone().detach().requires_grad_(True).type_as(y)

        # increment num evals
        self._num_evals += 1
        for state in states:
            state.requires_grad_(True)

        # sample and fix noise
        if self._e is None:
            self._e = torch.randn_like(y)
        
        with torch.set_grad_enabled(True):
            if len(states) == 3:
                c = states[2]
                tc = torch.cat([t, c.view(y.size(0), -1)], dim=-1)
                dy = self.diffeq(tc, y)
                divergence = self.divergence(dy, y, e=self._e).unsqueeze(-1)
                
                return dy, -divergence, torch.zeros_like(c).requires_grad_(True)
            
            elif len(states) == 2:
                dy = self.diffeq(t, y)
                divergence = self.divergence(dy, y, e=self._e).view(-1, 1)
                
                return dy, -divergence


class Decoder_Flow(nn.Module):
    def __init__(self, args):
        super(Decoder_Flow, self).__init__()

        self.set_cnf = build_model(args, args.input_dim, args.f_dims,
            args.z_dim, args.num_blocks, True)

    def forward(self, x, h, logp=True, reverse=False): # h: B x T x H  x: B x T x K x D
        # check the tensor shape
        assert x.dim() == 4
        assert h.dim() == 3
        assert x.size(1) == h.size(1)

        B, T, K, D = x.size()
        h_size = h.size(2)

        x = x.reshape(B*T, K, D)  # (B x T) x K x D
        h = h.reshape(B*T, h_size)

        if logp:
            x_0, delta_logpx = self.set_cnf(x, h, torch.zeros(*x.shape[:-1], 1).to(x), reverse=reverse)
            return x_0, delta_logpx
        else:
            x_0 = self.set_cnf(x, h, reverse=reverse)
            return x_0


def build_model(args, input_dim, hidden_dims, context_dim, num_blocks, conditional):
    def build_cnf():
        diffeq = ODEnet(
            hidden_dims=hidden_dims,
            input_shape=(input_dim, ),
            context_dim=context_dim,
        )
        odefunc = ODEfunc_Flow(
            diffeq=diffeq,
        )
        cnf = SetCNF(
            odefunc=odefunc,
            T=args.T_end,
            train_T=args.train_T,
            conditional=conditional,
            solver=args.solver,
            use_adjoint=args.use_adjoint,
            atol=args.atol,
            rtol=args.rtol
        )
        return cnf

    chain = [build_cnf() for _ in range(args.num_blocks)]
    model = SequentialFlow(chain)
    
    return model


class SequentialFlow(nn.Module):
    def __init__(self, layer_list):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layer_list)

    def forward(self, x, context, logpx=None, reverse=False, inds=None, integral_time=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain)-1, -1, -1)
            else:
                inds = range(len(self.chain))
        
        if logpx is None:
            for i in inds:
                x = self.chain[i](x, context, logpx, integral_time, reverse)
            return x
        else:
            for i in inds:
                x, logpx = self.chain[i](x, context, logpx, integral_time, reverse)
            return x, logpx


class SetCNF(nn.Module):
    def __init__(self, odefunc, conditional=True, T=1.0, train_T=False,
                 solver='dopri5', atol=1e-4, rtol=1e-3, use_adjoint=True):
        super(SetCNF, self).__init__()

        self.train_T = train_T
        self.T = T
        if train_T:
            self.register_parameter("sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T))))
        
        self.use_adjoint = use_adjoint
        self.odefunc = odefunc
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol
        self.solver_options = {}
        self.conditional = conditional
    
    def forward(self, x, context=None, logpx=None, integral_time=None, reverse=False):
        if logpx is None:
            _logpx = torch.zeros(*x.shape[:-1], 1).to(x)
        else:
            _logpx = logpx
        
        if self.conditional:
            assert context is not None
            states = (x, _logpx, context)
            atol = [self.atol] * 3
            rtol = [self.rtol] * 3
        else:
            states = (x, _logpx)
            atol = [self.atol] * 2
            rtol = [self.rtol] * 2
        
        if integral_time is None:
            if self.train_T:
                integral_time = torch.stack(
                    [torch.tensor(0.).to(x), self.sqrt_end_time * self.sqrt_end_time]
                ).to(x)
            else:
                integral_time = torch.tensor([0., self.T], requires_grad=False).float().to(x)
        
        if reverse:
            integral_time = _flip(integral_time, 0)
        
        self.odefunc.before_odeint()
        odeint = odeint_adjoint if self.use_adjoint else odeint_normal

        if self.training:
            state_t = odeint(
                self.odefunc,
                states,
                integral_time.to(x),
                atol=atol,
                rtol=rtol,
                method=self.solver,
                options=self.solver_options,
            )
        else:
            state_t = odeint(
                self.odefunc,
                states,
                integral_time.to(x),
                atol=atol,
                rtol=rtol,
                method=self.solver
            )
        
        if len(integral_time) == 2:
            state_t = tuple(s[1] for s in state_t)
        
        z_t, lopgz_t = state_t[:2]

        if logpx is not None:
            return z_t, lopgz_t
        else:
            return z_t
    
    def num_evals(self):
        return self.odefunc._num_evals.item()


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim)-1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class RNN_z0_Encoder(nn.Module):
    def __init__(self, h_dim, i_dim, rnn_dim, use_t=True):
        super(RNN_z0_Encoder, self).__init__()

        self.h_dim = h_dim
        self.i_dim = i_dim
        self.rnn_dim = rnn_dim
        self.use_t = use_t

        self.h_to_m = nn.Sequential(
            nn.Linear(self.rnn_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, self.h_dim)
        )

        self.h_to_v = nn.Sequential(
            nn.Linear(self.rnn_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, self.h_dim)
        )

        if self.use_t:
            self.i_dim += 1

        self.rnn = nn.GRU(self.i_dim, self.rnn_dim, batch_first=True)
    
    def forward(self, x, t, reverse=True):
        # x shape: [B, T, D]
        batch_size = x.size(0)
        T = x.size(1)

        if reverse:
            x = x[:,range(T-1, -1, -1),:]
        
        if self.use_t:
            delta_t = t[1:] - t[:-1]
            if reverse:
                l = delta_t.size(0)
                delta_t = delta_t[range(l-1, -1, -1)]
            
            delta_t = torch.cat((delta_t, torch.zeros(1).to(t)))
            delta_t = delta_t.unsqueeze(1).repeat((1,batch_size)).unsqueeze(-1).permute(1,0,2)
            x = torch.cat((delta_t, x), dim=-1)
        
        rnn_out, _ = self.rnn(x)
        h_z0 = rnn_out[:,-1,:]

        loc = self.h_to_m(h_z0)
        scale = self.h_to_v(h_z0)

        return loc, scale  # [B, D]


def create_net(n_inputs, n_outputs, n_layers = 1, 
	n_units = 256, nonlinear = nn.Tanh):
	layers = [nn.Linear(n_inputs, n_units)]
	for i in range(n_layers):
		layers.append(nonlinear())
		layers.append(nn.Linear(n_units, n_units))

	layers.append(nonlinear())
	layers.append(nn.Linear(n_units, n_outputs))
	return nn.Sequential(*layers)


class ODEFunc(nn.Module):
    def __init__(self, input_dim, h_dim):
        super(ODEFunc, self).__init__()

        self.ode_func = create_net(
            n_inputs=h_dim, n_outputs=h_dim, nonlinear=nn.Tanh
        )
        self.input_dim = input_dim

    def forward(self, t, x, reverse=False):
        dx = self.ode_func(x)

        if reverse:
            dx = - dx
        return dx


class ODE_z_Trans(nn.Module):
    def __init__(self, diffeq, h_dim, use_adjoint=True):
        super(ODE_z_Trans, self).__init__()

        self.ode_func = diffeq
        self.use_adjoint = use_adjoint
    
    def forward(self, z0, t, reverse=False):
        odeint = odeint_adjoint if self.use_adjoint else odeint_normal

        z_t = odeint(self.ode_func, z0, t, rtol=args.rtol, atol=args.atol, method=args.solver)
        z_t = z_t.permute(1, 0, 2)

        return z_t

    def sample_prior(self, z0, t, n_sample):
        pass


class Model(nn.Module):
    def __init__(self, set_encode, encode, trans, decode):
        super(Model, self).__init__()
        
        self.set_encoder = set_encode.to(device)
        self.decoder = decode.to(device)
        self.encoder = encode.to(device)
        self.trans = trans.to(device)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 50, 0.5)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=5, factor=0.2)
    
    def sample_gaussian(self, size, device=None):
        y = torch.randn(*size).float()
        y = y.to(device)
        return y

    def standard_normal_logprob(self, z):
        # dim = z.size(-1)
        log_z = -0.5 * log(2 * pi)
        return log_z - z.pow(2) / 2

    def kl_divergence(self, mu, logvar):
        kl = 0.5*(-1 - logvar + mu.pow(2) + logvar.exp())
        kl = kl.sum(1)
        return kl
    
    def reparam_sample(self, mean, logvar, N):
        std = torch.exp(0.5 * logvar)
        if N is not None:
            eps = Normal(torch.zeros_like(mean), torch.ones_like(mean)).sample([N,]).to(device)
        else:
            eps = torch.randn(std.size()).to(mean)
        z = mean + std * eps
        return z
        
    def elbo(self, x, t):
        h = self.set_encoder(x)
        mean, logvar = self.encoder(h, t) # mean: B * D
        z_0 = self.reparam_sample(mean, logvar, N=None)
        _z, delta_logpx = self.decoder(x, self.trans(z_0, t))

        logp_x = self.standard_normal_logprob(_z).view(x.size(0)*x.size(1), -1).sum(1, keepdim=True)
        delta_logpx = delta_logpx.view(x.size(0)*x.size(1), x.size(2), 1).sum(1)
        logp_x = logp_x - delta_logpx
        klds = self.kl_divergence(mean, logvar)
        
        elbo = logp_x.mean() - klds.mean()
        return elbo
    
    def encode(self, x, t):
        h = self.set_encoder(x)
        mu, logvar = self.encoder(h, t)
        return self.reparam_sample(mu, logvar, None)
    
    def decode(self, z_0, num_points, t):
        z = self.trans(z_0, t)
        y = torch.randn(z_0.size(0), len(t), num_points, args.input_dim).to(z)
        x = self.decoder(y, z, logp=False, reverse=True)
        return y, x
    
    def sample(self, t, batch_size, num_points, input_dim, z_dim, device):
        z_0 = self.sample_gaussian((batch_size, z_dim), device)
        z = self.trans(z_0, t)
        y = self.sample_gaussian((batch_size, len(t), num_points, input_dim), device)
        x = self.decoder(y, z, logp=False, reverse=True)
        return z, x
    
    def interpolate(self, t, z_0, batch_size, num_points, input_dim, device):
        z = self.trans(z_0, t)
        y = self.sample_gaussian((batch_size, len(t), num_points, input_dim), device)
        x = self.decoder(y, z, logp=False, reverse=True)
        return z, x
    
    def recons(self, x, t):
        num_points = x.size(2)
        z_0 = self.encode(x, t)
        _, x = self.decode(z_0, num_points, t)

        return x

    def save(self):
        fname = f'{args.exp_dir}/model.pth'
        torch.save(self.state_dict(), fname)        

    def load(self):
        fname = f'{args.exp_dir}/model.pth'
        self.load_state_dict(torch.load(fname))
        
    def set_data(self, train_loader, test_loader):
        self.train_loader = train_loader
        self.test_loader = test_loader
    
    def fit(self):
        best_train_ll = -np.inf
        best_test_ll = -np.inf
        for epoch in range(args.epochs):
            train_ll = self._train(epoch)
            test_ll = self.test(epoch)
            
            if train_ll > best_train_ll:
                best_train_ll = train_ll
            if test_ll > best_test_ll:
                best_test_ll = test_ll
                self.save()

            msg = f'{time.strftime("%H:%M:%S", time.localtime())} Epoch: {epoch} '
            msg += f'Train: {train_ll:.4f}/{best_train_ll:.4f} '
            msg += f'Test: {test_ll:.4f}/{best_test_ll:.4f}'

            logging.info(msg)

            self.scheduler.step(test_ll)
        
    def _train(self, epoch):
        self.train()
        train_ll = []
        for _, (t, x, _) in enumerate(tqdm(self.train_loader)):
            x = x.to(device)
            t = t[0].to(device)
            self.optimizer.zero_grad()
            elbo = self.elbo(x, t)
            loss = -elbo
            loss.backward()
            self.optimizer.step()
            train_ll.append(elbo.data.cpu().numpy())
        train_ll = np.mean(train_ll)

        samples = model.recons(x, t)
        samples = samples.reshape(*x.size())[0,:,:,:]
        results = []
        for idx in range(min(10, x.size(1))):
            res = visualize_point_clouds(samples[idx], x[0, idx], idx)
            results.append(res)
        res = np.concatenate(results, axis=1)
        imageio.imwrite(os.path.join(args.exp_dir, 'tr_vis_trainconditioned_epoch%d-gpu%s.png' %(epoch, args.gpu)), res.transpose((1, 2, 0)))
        
        return train_ll

    def test(self, epoch):
        self.eval()
        test_ll = []
        with torch.no_grad():
            for _, (t, x, _) in enumerate(tqdm(self.test_loader)):
                x = x.to(device)
                t = t[0].to(device)
                elbo = self.elbo(x, t)
                test_ll.append(elbo.data.cpu().numpy())
        test_ll = np.mean(test_ll)
        
        samples = model.recons(x, t)
        samples = samples.reshape(*x.size())[0,:,:,:]
        results = []
        for idx in range(min(10, x.size(1))):
            res = visualize_point_clouds(samples[idx], x[0, idx], idx)
            results.append(res)
        res = np.concatenate(results, axis=1)
        imageio.imwrite(os.path.join(args.exp_dir, 'tr_vis_testconditioned_epoch%d-gpu%s.png' %(epoch, args.gpu)), res.transpose((1, 2, 0)))

        return test_ll


def visualize_point_clouds(pts, gtr, idx, pert_order=[0, 1]):
    pts = pts.cpu().detach().numpy()[:, pert_order]
    gtr = gtr.cpu().detach().numpy()[:, pert_order]

    fig = plt.figure(figsize=(6, 3))
    ax1 = fig.add_subplot(121)
    ax1.set_title("Sample:%s" % idx)
    ax1.scatter(pts[:, 0], pts[:, 1], s=5)

    ax2 = fig.add_subplot(122)
    ax2.set_title("Ground Truth:%s" % idx)
    ax2.scatter(gtr[:, 0], gtr[:, 1], s=5)

    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    res = np.array(fig.canvas.renderer._renderer)
    res = np.transpose(res, (2, 0, 1))

    plt.close()
    return res


def visualize_batch(t, pts, shape='2d', exp_dir='./exp', name='sample'):
    if shape not in ['2d', '3d']:
        raise NotImplementedError('the shape should be in 2d or 3d')
    t = t.cpu().detach().numpy()
    pts = pts.cpu().detach().numpy()

    if shape == '3d':
        fig = plt.figure(figsize=(t.shape[0]*2, pts.shape[0]*2), projection='3d')
    else:
        fig = plt.figure(figsize=(t.shape[0]*2, pts.shape[0]*2))
    for idx in range(pts.shape[0]):
        for j in range(t.shape[0]):
            ax = fig.add_subplot(pts.shape[0], t.shape[0], idx*t.shape[0]+j+1)
            ax.scatter(pts[idx,j,:,0], pts[idx,j,:,1], s=10)
            ax.set_xlim(-.8, .8)
            ax.set_ylim(-.8, .8)
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.set_title(f'sample: {idx}, t={t[j]:.2f}')
            if idx == 0:
                ax.set_title(f't={t[j]:.2f}', fontsize=25)
    plt.tight_layout()
    plt.savefig(f'{exp_dir}/{name}.png', dpi=300)
    plt.close('all')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    decoder = Decoder_Flow(args)
    set_encoder = SetEncoder(args.input_dim, args.en_dims)
    encoder = RNN_z0_Encoder(args.z_dim, args.en_dims[-1], args.rnn_dim, True)
    ode_func = ODEFunc(args.input_dim, args.z_dim)
    trans = ODE_z_Trans(ode_func, args.z_dim, True)

    model = Model(set_encoder, encoder, trans, decoder)
    print(model)
    print(f"The model has the number of parameters of {count_parameters(model)}")

    train_loader, test_loader = get_loader(is_time_varying=True, data_dir='./data', batch_size=args.batch_size, num_times=args.T)
    model.set_data(train_loader, test_loader)

    # train model
    model.fit()





    


    

