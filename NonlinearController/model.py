import deepSI
from deepSI import System_data, System_data_list
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import optim, nn
from tqdm.auto import tqdm
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from deepSI.utils import simple_res_net

class Matrix_NN(nn.Module):
    def __init__(self, nx, nu, n_rows, n_columns, NN=simple_res_net, NN_kwargs=dict(), norm=1):
        #nu is False to indicate not a dependency on u
        super().__init__()
        self.nx = nx
        self.no_u = nu is False
        self.nu = nu
        self.nu_val = 0 if self.no_u else (1 if nu is None else nu)
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.net = NN(n_in=nx + self.nu_val, n_out=self.n_rows*self.n_columns, **NN_kwargs)
        self.norm = norm*self.n_columns**0.5
    
    def forward(self, x, u=None):
        if self.no_u:
            net_in = x
        else:
            net_in = torch.cat([x, u.reshape(u.shape[0], -1)], dim=1)
        net_out = self.net(net_in).reshape(x.shape[0], self.n_rows, self.n_columns)
        return net_out/self.norm


class Velocity_from_net(nn.Module):
    def __init__(self, nx, nu, ny, feedthrough=False, F_net=Matrix_NN, F_net_kwargs={'norm':10}, \
                                                      G_net=Matrix_NN, G_net_kwargs={'norm':10},\
                                                      H_net=Matrix_NN, H_net_kwargs={},):
        super(Velocity_from_net, self).__init__()
        assert feedthrough==False
        self.nx = nx
        self.nu = nu
        self.nu_val = 1 if nu is None else nu
        self.ny = ny
        self.ny_val = 1 if ny is None else ny
        #nx, ny, nu=-1,
        self.F = F_net(nx, nu, n_rows=nx, n_columns=nx,                 **F_net_kwargs)
        self.G = G_net(nx, nu, n_rows=nx, n_columns=self.nu_val,        **G_net_kwargs)
        self.H = H_net(nx, False, n_rows=self.ny_val, n_columns=self.nx,**H_net_kwargs)

    def forward(self, X, u):
        x = X[:,:self.nx]
        dx = X[:,self.nx:self.nx*2]
        um = X[:,self.nx*2:self.nx*2+self.nu_val]
        ym = X[:,-self.ny_val:]
        
        u = u.reshape(u.shape[0],-1)
        
        y = ym + torch.einsum('bij,bj->bi', self.H(x), dx)
        xp = x + \
              torch.einsum('bij,bj->bi',self.F(x, u), dx) + \
              torch.einsum('bij,bj->bi',self.G(x, u), u - um)
        
        dxp = xp - x
        xnext = torch.cat([xp, dxp, u, y], dim=1)
        return (y[:,0] if self.ny==None else y), xnext
    
class Velocity_form_encoder(nn.Module):
    def __init__(self, nb, nu, na, ny, nx, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh, dx_norm=20):
        super(Velocity_form_encoder, self).__init__()
        from deepSI.utils import simple_res_net
        self.nu_tuple = tuple() if nu is None else ((nu,) if isinstance(nu,int) else nu)
        self.ny_tuple = tuple() if ny is None else ((ny,) if isinstance(ny,int) else ny)
        
        self.net = simple_res_net(n_in=nb*np.prod(self.nu_tuple,dtype=int) + na*np.prod(self.ny_tuple,dtype=int), \
            n_out=nx*2, n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers, activation=activation)
        self.norm = torch.ones(nx*2)
        self.norm[nx:] /= dx_norm

    def forward(self, upast, ypast):
        net_in = torch.cat([upast.view(upast.shape[0],-1),ypast.view(ypast.shape[0],-1)],axis=1)
        xdx = self.net(net_in)*self.norm[None]
        return torch.cat([xdx, upast[:,-1:], ypast[:,-1:]], dim=1)
    
from deepSI.fit_systems import SS_encoder_general_hf
    
class SS_encoder_velocity_from(SS_encoder_general_hf):
    """The encoder function with combined h and f functions
    
    the hf_net_default has the arguments
       hf_net_default(nx, nu, ny, feedthrough=False, **hf_net_kwargs)
    and is used as 
       ynow, xnext = hfn(x,u)
    """
    def __init__(self, nx=10, na=20, nb=20, feedthrough=False, \
                 hf_net=Velocity_from_net, \
                 hf_net_kwargs = dict(), \
                 e_net=Velocity_form_encoder,   e_net_kwargs={}, na_right=0, nb_right=0):
        assert na_right==0 and nb_right==0, 'not yet implemented for nonzero na_right, nb_right'
        assert feedthrough==False
        super(SS_encoder_velocity_from, self).__init__(nx=nx, nb=nb, na=na, na_right=na_right, nb_right=nb_right, \
                                                      hf_net=hf_net, hf_net_kwargs=hf_net_kwargs, e_net=e_net, e_net_kwargs=e_net_kwargs )
        self.nx0 = nx
    def init_nets(self, nu, ny): # a bit weird
        na_right = self.na_right if hasattr(self,'na_right') else 0
        nb_right = self.nb_right if hasattr(self,'nb_right') else 0
        nu_val = 1 if nu is None else (nu if isinstance(nu, int) else np.prod(nu, dtype=int))
        ny_val = 1 if ny is None else (ny if isinstance(ny, int) else np.prod(ny, dtype=int))
        self.nx = self.nx0*2 + nu_val + ny_val
        
        self.encoder = self.e_net(nb=self.nb+nb_right, nu=nu, na=self.na+na_right, ny=ny, nx=self.nx0, **self.e_net_kwargs)
        self.hfn = self.hf_net(nx=self.nx0, nu=nu, ny=ny, **self.hf_net_kwargs)

sys = SS_encoder_velocity_from(nx=2, na=5, nb=5)
train, test = deepSI.datasets.Silverbox()
train, val = train.train_test_split(0.1)

sys.fit(train, val, loss_kwargs=dict(nf=80))
sys.save_system('velocity-form/silverbox-test-sys')