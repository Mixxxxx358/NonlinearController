import deepSI
import numpy as np
import torch
from torch import nn
import systems

# change output from linear layer to ny=2 (from sin cos output), to sin and cos applied to linear layer to ny=1
    #!!! There is probably an issue with the input of the sin and cos being normalized
class sincos_output_net(nn.Module):
    def __init__(self, nx, ny, nu=-1, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(sincos_output_net, self).__init__()
        from deepSI.utils import simple_res_net
        self.ny = tuple() if ny is None else ((ny,) if isinstance(ny,int) else ny)
        self.feedthrough = nu!=-1
        if self.feedthrough:
            self.nu = tuple() if nu is None else ((nu,) if isinstance(nu,int) else nu)
            net_in = nx + np.prod(self.nu, dtype=int)
        else:
            net_in = nx
        self.net = simple_res_net(n_in=net_in, n_out=np.prod((1,),dtype=int), n_nodes_per_layer=n_nodes_per_layer, \
            n_hidden_layers=n_hidden_layers, activation=activation)

    def forward(self, x, u=None):
        xu = x
        xu = self.net(xu).view(*((x.shape[0],)+(1,)))
        y = torch.cat([torch.sin(xu), torch.cos(xu)], dim=1)
        return y
    
# change input from sin cos to arctan2 of sin cos.
    #!!! There is probably a problem with the sin and cos being normalized at the arctan2
class arctan2_encoder_net(nn.Module):
    def __init__(self, nb, nu, na, ny, nx, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(arctan2_encoder_net, self).__init__()
        from deepSI.utils import simple_res_net
        self.nu = tuple() if nu is None else ((nu,) if isinstance(nu,int) else nu)
        self.ny = tuple() if ny is None else ((ny,) if isinstance(ny,int) else ny)
        self.net = simple_res_net(n_in=nb*np.prod(self.nu,dtype=int) + na*np.prod((1,),dtype=int), \
            n_out=nx, n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers, activation=activation)

    def forward(self, upast, ypast):
        ypast = torch.mul(torch.atan2(ypast[:,:,0],ypast[:,:,1]),0.5)
        net_in = torch.cat([upast.view(upast.shape[0],-1),ypast.view(ypast.shape[0],-1)],axis=1)
        return self.net(net_in)

class SinCos_encoder(deepSI.fit_systems.SS_encoder_general):
    def __init__(self, nx=10, na=20, nb=20, na_right=0, nb_right=0, e_net_kwargs={}, f_net_kwargs={}, h_net_kwargs={}):
        super(SinCos_encoder, self).__init__(nx=nx, na=na, nb=nb, na_right=na_right, nb_right=nb_right, e_net_kwargs=e_net_kwargs, f_net_kwargs=f_net_kwargs, h_net_kwargs=h_net_kwargs)
        # self.h_net = sincos_output_net
        self.e_net = arctan2_encoder_net
        # self.norm.y0 = np.array([0.0,0.0])
        # self.norm.ystd = np.array([1.0,1.0])
        print(self.norm.ystd)

    def init_nets(self, nu, ny): # a bit weird
        na_right = self.na_right if hasattr(self,'na_right') else 0
        nb_right = self.nb_right if hasattr(self,'nb_right') else 0
        self.encoder = self.e_net(nb=(self.nb+nb_right), nu=nu, na=(self.na+na_right), ny=ny, nx=self.nx, **self.e_net_kwargs)
        self.fn =      self.f_net(nx=self.nx, nu=nu,                                **self.f_net_kwargs)
        if self.feedthrough:
            self.hn =      self.h_net(nx=self.nx, ny=ny, nu=nu,                     **self.h_net_kwargs) 
        else:
            self.hn =      self.h_net(nx=self.nx, ny=ny,                            **self.h_net_kwargs)

    def init_model(self, sys_data=None, nu=-1, ny=-1, device='cpu', auto_fit_norm=True, optimizer_kwargs={}, parameters_optimizer_kwargs={}, scheduler_kwargs={}):
        '''This function set the nu and ny, inits the network, moves parameters to device, initilizes optimizer and initilizes logging parameters'''
        if sys_data==None:
            assert nu!=-1 and ny!=-1, 'either sys_data or (nu and ny) should be provided'
            self.nu, self.ny = nu, ny
        else:
            self.nu, self.ny = sys_data.nu, sys_data.ny
            # if auto_fit_norm:
            #     self.norm.fit(sys_data)
            self.norm.ustd = 2.5
        self.init_nets(self.nu, self.ny)
        self.to_device(device=device)
        parameters_and_optim = [{**item,**parameters_optimizer_kwargs.get(name,{})} for name,item in self.parameters_with_names.items()]
        self.optimizer = self.init_optimizer(parameters_and_optim, **optimizer_kwargs)
        self.scheduler = self.init_scheduler(**scheduler_kwargs)
        self.bestfit = float('inf')
        self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        self.init_model_done = True


# Nu = 25
# a = 3; flips = 2000
# u = np.random.normal(0,1.0,Nu)
# for i in range(flips-1):
#     u = np.hstack((u, np.random.normal(a,.7,Nu)))
#     a = -1*a
#     u = np.hstack((u, np.random.normal(0,.7,Nu)))

# Nu = 200000
# a = 5.0
# u = deepSI.deepSI.exp_design.multisine(Nu, pmax=Nu//2-1, n_crest_factor_optim=20)*a/2
# u = np.clip(u, -a, a)

# Nu = 800000
# a = 2.6
# u = deepSI.deepSI.exp_design.multisine(Nu, pmax=Nu//2-1, n_crest_factor_optim=5)*0.8
# flips = 1000
# for i in range(flips):
#     u[i*(Nu//flips):(i+1)*(Nu//flips)] = u[i*(Nu//flips):(i+1)*(Nu//flips)] + a
#     a = -1*a

# u = np.load("NonlinearController/data/physical_NMPC_w.npy")[0,:]
# y = np.load("NonlinearController/data/physical_NMPC_q.npy").T

# u = np.load("NonlinearController/data/PidSetPointsR3_8.npy")
# u = np.load("NonlinearController/data/Energy.npy")
# a = 5.0

a = 6.0
u = np.load("NonlinearController/data/Energy_u.npy")
y = np.load("NonlinearController/data/Energy_y.npy")
y = np.vstack((np.sin(y),np.cos(y))).T

sigma_n = [0.01]; dt = 0.05
# setup = systems.SinCosUnbalancedDisc(dt=dt, sigma_n=sigma_n)
# data = setup.apply_experiment(deepSI.System_data(u=u))
data = deepSI.System_data(u=u, y=y)

train, val = data.train_test_split(split_fraction=0.4)
# train, val = train.train_test_split(split_fraction=0.25)

n_nodes_per_layer = 64; n_hidden_layers = 2
net_params = {'n_nodes_per_layer':n_nodes_per_layer, 'n_hidden_layers':n_hidden_layers}
# sys = deepSI.fit_systems.SS_encoder_general(nx=3, na=24, nb=24, na_right=1, \
#                                             f_net_kwargs=net_params, h_net_kwargs=net_params, e_net_kwargs=net_params)

sys = SinCos_encoder(nx=2, na=12, nb=12, na_right=1, \
                    f_net_kwargs=net_params, h_net_kwargs=net_params, e_net_kwargs=net_params)

epochs = 40; batch_size=128; nf=10
sys.fit(train_sys_data=train, val_sys_data=val, epochs=epochs, batch_size=batch_size, loss_kwargs={'nf':nf})

addres = 'NonlinearController/trained_models/sincos/'
name = addres + "arctan_dt" + str(dt).replace(".", "_") + "_e" + str(epochs) + "_b" + str(batch_size) + "_nf" + str(nf) + "_amp" + str(a).replace(".", "_") + "_sn" + str(sigma_n[0]).replace(".", "_")
sys.save_system(name)
# sys.save_system('NonlinearController/trained_models/sincos/sincos-test')