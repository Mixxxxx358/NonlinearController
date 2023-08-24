from NonlinearController.model_utils import *
from NonlinearController.systems import UnbalancedDisc
from matplotlib import pyplot as plt
import deepSI
from casadi import *
import torch
import random
import time

##################  Utility functions  #######################

def randomLevelReference(Nsim, nt_range, level_range):
    x_reference_list = np.array([])
    Nsim_remaining = Nsim
    while True:
        Nsim_steps = random.randint(nt_range[0],nt_range[1])
        Nsim_remaining = Nsim_remaining - Nsim_steps
        x_reference_list = np.hstack((x_reference_list, np.ones(Nsim_steps)*random.randint(level_range[0]*10,level_range[1]*10)/10))

        if Nsim_remaining <= 0:
            x_reference_list = x_reference_list[:Nsim]
            break
    return x_reference_list

def setPointInput(y_ref):
    g = 9.80155078791343
    J = 0.000244210523960356
    Km = 10.5081817407479
    I = 0.0410772235841364
    M = 0.0761844495320390
    tau = 0.397973147009910

    return (tau * M * g * I)/(Km * J) * np.sin(y_ref)

##################  System  #######################
dt = 0.1
system = UnbalancedDisc(dt=dt)
system.reset_state()

##################  MPC variable specification  #######################
model = deepSI.load_system("NonlinearController/trained_models/unbalanced/ObserverUnbalancedDisk_dt01_nab_4_SNR_30_e250")
# model = deepSI.load_system("NonlinearController/trained_models/unbalanced/ud_test_4")
Nc=5; nr_sim_steps = 100

Q = 100; R = 1

w_min = -4.0; w_max = 4.0
q_min = [-1.2]; q_max = [1.2]
w0 = 0; q0 = 0

a = 0.9; reference = np.hstack((np.ones(20)*a,np.ones(20)*-a,np.ones(20)*a/2,np.ones(20)*-a/2,np.ones(40)*0))
reference = np.load("references/multisine.npy")

# reference = randomLevelReference(nr_sim_steps+Nc, [25,30], [-1,1])
# reference = deepSI.deepSI.exp_design.multisine(nr_sim_steps+Nc+1, pmax=20, n_crest_factor_optim=20)/2.0
# reference = np.load("NonlinearController/references/setPoints.npy")
# reference = np.sin(np.arange(0,nr_sim_steps+Nc)/np.pi*3.5)*1
# x_reference_list = 1*np.load("NonlinearController/references/randomLevelTime25_30Range-1_1Nsim500.npy")
# reference = x_reference_list[1,:]

##################  Offline Computation  #######################
nx = model.nx
x = MX.sym("x",nx,1)
nu = model.nu if model.nu is not None else 1
u = MX.sym("u",nu,1)
ny = model.ny if model.ny is not None else 1

# convert torch nn to casadi function
x_rhs = CasADi_Fn(model, x, u)
y_rhs = CasADi_Hn(model, x)

f = Function('f', [x, u], [x_rhs])
h = Function('h', [x], [y_rhs])

# normalize initial input and output
norm = model.norm
u0 = norm_input(w0, norm)
y0 = norm_output(q0, norm)

u_min = norm_input(w_min, norm); u_max = norm_input(w_max, norm)
y_min = norm_output(q_min, norm); y_max = norm_output(q_max, norm)

# initialize observer history input and output
nb = model.nb
uhist = torch.ones((1,nb))*u0
na = model.na
yhist = torch.ones((1,na+1))*y0

# define initial predicted states and inputs
X0 = np.tile(model.encoder(uhist,yhist).detach().numpy().T,Nc+1)
U0 = np.ones((Nc)*nu)[np.newaxis]*u0
Y0 = np.ones((Nc)*ny)[np.newaxis]*y0

# define opti stack
opti  = Opti()

states = opti.variable(nx, Nc+1)
controls = opti.variable(nu, Nc)
outputs = opti.variable(ny, Nc)

x_initial = opti.parameter(nx, 1)
y_ref = opti.parameter(ny, Nc)
u_ref = opti.parameter(nu, Nc)

opti.subject_to(opti.bounded(y_min,outputs,y_max))
opti.subject_to(opti.bounded(u_min,controls,u_max))
opti.subject_to(states[:,0] == x_initial)

opti.set_initial(states, X0)
opti.set_initial(controls, U0)
opti.set_initial(outputs, Y0)

opti.set_value(x_initial,X0[:,0])
# opti.set_value(y_ref, reference[:Nc])

opts = {'print_time' : 0, 'ipopt': {'print_level': 0}}
opti.solver("ipopt",opts)

objective = 0
for i in np.arange(Nc):
    opti.subject_to(states[:,i+1] == f(states[:,i], controls[:,i]))
    opti.subject_to(outputs[:,i] == h(states[:,i+1]))
    objective = (objective + 
                    mtimes(mtimes((outputs[:,i] - y_ref[:,i]).T,Q),(outputs[:,i] - y_ref[:,i])) +
                    mtimes(mtimes((controls[:,i] - u_ref[:,i]).T,R),(controls[:,i] - u_ref[:,i])))

opti.minimize(objective)

##################  Logging  #######################
log_q = np.zeros((ny,nr_sim_steps))
log_w = np.zeros((nu,nr_sim_steps))

log_comp_t = np.zeros((2, nr_sim_steps))

##################  Online Computation  #######################

#++++++++++++++++++ start simulation step +++++++++++++++++++++++
for k in range(nr_sim_steps):
    component_start = time.time()
    
    opti.set_value(y_ref, norm_output(reference[k:k+Nc], norm))
    opti.set_value(u_ref, norm_input(setPointInput(reference[k:k+Nc]), norm))

    log_comp_t[0, k] = log_comp_t[0, k] + time.time() - component_start
    component_start = time.time()

    sol = opti.solve()

    log_comp_t[1, k] = log_comp_t[1, k] + time.time() - component_start
    component_start = time.time()

    U0[0,:] = sol.value(controls)
    X0[:,:] = sol.value(states)
    Y0[:,:] = sol.value(outputs)

    # determine input from optimal velocity input
    u0 = U0[:,0]
    # denormalize input
    w0 = denorm_input(u0, norm)
    # measure output then apply input
    system.x = system.f(system.x, w0[0])
    q1 = system.h(system.x, w0[0])
    # normalize output
    y1 = norm_output(q1, norm)

    # shift history input and output for encoder
    for j in range(nb-1):
        uhist[0,j] = uhist[0,j+1]
    uhist[0,nb-1] = torch.Tensor(u0)
    for j in range(na):
        yhist[0,j] = yhist[0,j+1]
    yhist[0,na] = torch.Tensor([y1])
    # predict state with encoder
    x1 = model.encoder(uhist,yhist)

    # shift predicted states, input, and output one time step k
    X0[:, :-1] = X0[:, 1:]; X0[:, -1:] = X0[:, -2:-1]; X0[:, :1] = x1.detach().numpy().T
    U0[:, :-1] = U0[:, 1:]; U0[:, -1:] = U0[:, -2:-1]#; U0[:, :1] = u0
    Y0[:, :-1] = Y0[:, 1:]; Y0[:, -1:] = Y0[:, -2:-1]; Y0[:, :1] = y1

    opti.set_initial(states, X0)
    opti.set_initial(controls, U0)
    opti.set_initial(outputs, Y0)
    opti.set_value(x_initial,X0[:,0])

    # log system signals
    log_q[:,k] = q1
    log_w[:,k] = w0

    log_comp_t[0, k] = log_comp_t[0, k] + time.time() - component_start

    # print progress
    # print("Sim step: " + str(k))
    
#++++++++++++++++++ end simulation step +++++++++++++++++++++++

fig1 = plt.figure(figsize=[12, 8])

plt.subplot(1,2,1)
plt.plot(np.arange(nr_sim_steps)*dt, log_w[0,:], label='input')
plt.plot(np.arange(nr_sim_steps)*dt, np.ones(nr_sim_steps)*w_max, 'r-.')#, label='max')
plt.plot(np.arange(nr_sim_steps)*dt, np.ones(nr_sim_steps)*w_min, 'r-.')#, label='min')
# plt.xlabel("time [s]")
plt.ylabel("voltage [V]")
plt.grid()
plt.legend(loc='upper right')

plt.subplot(1,2,2)
plt.plot(np.arange(nr_sim_steps)*dt, log_q[0,:], label='output')
plt.plot(np.arange(nr_sim_steps)*dt, reference[:nr_sim_steps], '--', label='reference')
plt.plot(np.arange(nr_sim_steps)*dt, np.ones(nr_sim_steps)*q_max[0], 'r-.')#, label='max')
plt.plot(np.arange(nr_sim_steps)*dt, np.ones(nr_sim_steps)*q_min[0], 'r-.')#, label='min')
# plt.xlabel("time [s]")
plt.ylabel("angle [rad]")
plt.grid()
plt.legend(loc='upper right')

plt.show()

CT_iters = np.split(log_comp_t, nr_sim_steps, axis=1)
CT = np.sum(CT_iters[0], axis=1)

remove_start = 0
S_iter = np.zeros(nr_sim_steps-remove_start)
T_iter = np.zeros(nr_sim_steps-remove_start)

for i in range(remove_start,nr_sim_steps):
    CT = np.sum(CT_iters[i], axis=1)
    S_iter[i-remove_start] = CT[1]
    T_iter[i-remove_start] = np.sum(CT)

Sorted = np.sort(T_iter)
# np.max(T_iter)*1000, np.mean(Sorted[int(nr_sim_steps*0.95):])*1000, np.mean(T_iter)*1000, np.std(T_iter)*1000, np.mean(S_iter)*1000 #in ms
Times = [np.max(T_iter)*1000,  np.mean(T_iter)*1000, np.std(T_iter)*1000, np.mean(S_iter)*1000] #in ms
print(Times)

# np.save("experiments/ud_nmpc_encoder_levels_u.npy", log_w)
# np.save("experiments/ud_nmpc_encoder_levels_q.npy", log_q)

np.save("experiments/ud_nmpc_encoder_sinus_u.npy", log_w)
np.save("experiments/ud_nmpc_encoder_sinus_q.npy", log_q)