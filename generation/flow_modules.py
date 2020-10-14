import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint_adjoint as odeint

class Flow(nn.Module):
    def __init__(self, args):
        super(Flow, self).__init__()

        self.trans = ODETrans(args.num_blocks, args.chn_size, args.dims, args.T, args.tol, args.solver, args.steps)
        self.likel = Gaussian(args.device)

    def logp(self, x):
        z, ldet = self.trans(x)
        logpz = self.likel.logp(z)
        logpx = logpz + ldet

        return logpx

    def sample(self, shape):
        z_sam = self.likel.sample(shape)
        x_sam, _ = self.trans(z_sam, reverse=True)

        return x_sam

# Peq transformations

### ODE trans
def divergence_fn(f, x, e):
    e_dzdx = torch.autograd.grad(f, x, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(x.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx

def FCnet(in_dim, h_dims):
    net = []
    for h in h_dims[:-1]:
        net.append(nn.Linear(in_dim, h))
        net.append(nn.Tanh())
        in_dim = h
    net.append(nn.Linear(in_dim, h_dims[-1]))

    return nn.Sequential(*net)

class ODEnet(nn.Module):
    def __init__(self, i_dim, f_dim, num_head=4):
        super(ODEnet, self).__init__()
        self.dim = f_dim[-1]
        self.num_head = num_head
        self.K = FCnet(i_dim, f_dim)
        self.Q = FCnet(i_dim, f_dim)
        self.V = FCnet(i_dim, f_dim)
        self.M = nn.Linear(f_dim[-1], i_dim)

    def forward(self, t, x):
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

class ODEfunc(nn.Module):
    def __init__(self, diffeq):
        super(ODEfunc, self).__init__()

        self.diffeq = diffeq
        self.register_buffer("_num_evals", torch.tensor(0.))
    
    def before_odeint(self):
        self._e = None
        self._num_evals.fill_(0)
    
    def num_evals(self):
        return self._num_evals.item()

    def forward(self, t, states):
        assert len(states) == 2, 'need x(t) and logpx(t)'
        x = states[0]
        
        # increment num evals
        self._num_evals += 1

        # convert to tensor
        t = torch.tensor(t).type_as(x)

        # Sample and fix the noise
        if self._e is None:
            self._e = torch.randn_like(x)

        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            t.requires_grad_(True)
            dx = self.diffeq(t, x)
            divergence = divergence_fn(dx, x, self._e)

        return (dx, -divergence)

def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class CNF(nn.Module):
    def __init__(self, odefunc, T, tol, solver, steps):
        super(CNF, self).__init__()

        self.register_parameter('sqrt_end_time', nn.Parameter(torch.sqrt(torch.tensor(T))))
        self.odefunc = odefunc
        self.atol = tol
        self.rtol = tol
        self.solver = solver
        self.steps = steps

    def forward(self, z, logpz, reverse=False):
        if self.solver in ['euler', 'rk4']:
            integration_times = torch.linspace(0.0, self.sqrt_end_time * self.sqrt_end_time, self.steps).to(z)
        else:
            integration_times = torch.tensor([0.0, self.sqrt_end_time * self.sqrt_end_time]).to(z)
        
        if reverse:
            integration_times = _flip(integration_times, 0)

        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()

        state_t = odeint(self.odefunc, (z, logpz), integration_times, atol=self.atol, rtol=self.rtol, method=self.solver)

        if len(integration_times) >= 2:
            state_t = tuple(s[-1] for s in state_t)

        z_t, logpz_t = state_t

        return z_t, logpz_t

    def num_evals(self):
        return self.odefunc._num_evals.item()

class ODETrans(nn.Module):
    def __init__(self, num_blocks, chn_size, dims, T, tol, solver, steps):
        super(ODETrans, self).__init__()

        chain = []
        for i in range(num_blocks):
            diffeq = ODEnet(chn_size, dims)
            odefunc = ODEfunc(diffeq)
            cnf = CNF(odefunc, T, tol, solver, steps)
            chain.append(cnf)
        self.chain = nn.ModuleList(chain)

    def forward(self, x, reverse=False):
        if reverse:
            inds = range(len(self.chain) - 1, -1, -1)
        else:
            inds = range(len(self.chain))

        logpx = torch.zeros(x.shape[0]).to(x)
        for i in inds:
            x, logpx = self.chain[i](x, logpx, reverse=reverse)
        
        return x, -logpx

# base likelihood

### indepentent Gaussian
class Gaussian(nn.Module):
    def __init__(self, device):
        super(Gaussian, self).__init__()

        self.device = device

    def logp(self, z):
        logZ = -0.5 * np.log(2 * np.pi)
        log_likel = logZ - z.pow(2) / 2
        log_likel = log_likel.sum(2).sum(1)

        return log_likel

    def sample(self, shape):
        return torch.randn(*shape).to(self.device)

class MultiGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(MultiGRUCell, self).__init__()
        cells = []
        for h in hidden_size:
            cells.append(nn.GRUCell(input_size, h, bias))
            input_size = h
        self.cells = nn.ModuleList(cells)
        self.num_cells = len(hidden_size)

    def forward(self, x, hx=None):
        if hx is None:
            hx = [None] * self.num_cells

        state = []
        for i in range(self.num_cells):
            x = self.cells[i](x, hx[i])
            state.append(x)
        
        return state

### independent autoregressive
class AutoReg(nn.Module):
    def __init__(self, device, num_layers, num_units):
        super(AutoReg, self).__init__()

        self.device = device

        self.cell = MultiGRUCell(1, [num_units]*num_layers)
        self.proj = nn.Linear(num_units, 2)

    def logp(self, z):
        B,N,d = z.shape
        z = z.view([B*N, d])
        params = []
        zi = z.new_ones([B*N, 1]) * -1
        hx = None
        for i in range(d):
            hx = self.cell(zi, hx)
            p = self.proj(hx[-1])
            params.append(p)
            zi = z[:,i,None]
        params = torch.stack(params, dim=1)
        mean, logs = params[:,:,0], params[:,:,1]

        log_likel = -0.5*np.log(2.*np.pi) - logs - 0.5*(z-mean).pow(2)/torch.exp(2.*logs)
        log_likel = log_likel.view([B,N,d]).sum(2).sum(1)

        return log_likel

    def sample(self, shape):
        B,N,d = shape
        sam = []
        zi = torch.ones([B*N, 1]).to(self.device) * -1 
        hx = None
        for i in range(d):
            hx = self.cell(zi, hx)
            p = self.proj(hx[-1])
            pm, ps = p[:,0], torch.exp(p[:,1])
            s = torch.randn([B*N]).to(self.device) * ps + pm
            sam.append(s)
        sam = torch.stack(sam, dim=1).view([B,N,d])

        return sam



