import numpy as np
import matplotlib.pyplot as plt
import time
import deepSI
from NonlinearController.utils import randomLevelReferenceSteps, wrapDisc, randomLevelReference, rmse
from NonlinearController.controllers import VelocityMpcController, controlLoopTime
from NonlinearController.systems import FullMassSpringDamper
from NonlinearController.models import CasADi_model, odeCasADiUnbalancedDisc

##################  System  #######################
dt = 0.01
k=1e4; c=5; m=0.25
system = FullMassSpringDamper(k, c, m, dt=dt, sigma_n=[0])#1e-4
system.reset_state()
nu = system.nu if system.nu is not None else 1
ny = system.ny if system.ny is not None else 1

##################  Reference  #######################
a = 3; reference_x = np.hstack((np.ones(20)*a,np.ones(20)*-a,np.ones(20)*a/2,np.ones(20)*-a/2,np.ones(40)*0))*1e-3
# reference_x = deepSI.deepSI.exp_design.multisine(nr_sim_steps+Nc, pmax=15, n_crest_factor_optim=10)*4e-3
# reference_x = np.sin(np.arange(0,nr_sim_steps+Nc)/np.pi*2.5)*6e-3
reference = reference_x[np.newaxis]

##################  Start Loop  #######################
fig1 = plt.figure(figsize=[8.9, 8])
for nx in range(2,9,2):

    ##################  MPC controller parameters  #######################
    Nc=10; max_iter = 1; nr_sim_steps = 100
    wlim = 60
    qlim = 1e-2
    nz = nx+ny; ne = 1

    Q1 = np.zeros((ny,ny)); np.fill_diagonal(Q1, [5])
    Q2 = np.zeros((nz,nz)); Q2[ny:,ny:] = np.eye(nx)*1
    R = np.eye(nu)*1
    P = np.eye(ny)*0.01

    ##################  Model  #######################
    base_name = "NonlinearController/trained_models/MSD/msd_nx"
    model = deepSI.load_system(base_name + str(nx))

    ##################  Control Loop MVT  #######################
    log_w, log_q, log_comp_t = controlLoopTime(reference_x, system, model, Nc, nr_sim_steps, nu, ny, Q1, Q2, R, P, qlim, wlim, max_iter, 5, 3, "LPV")

    ##################  Plots  #######################
    plt.plot(np.arange(nr_sim_steps)*dt, log_q[0,:], label=str(nx))
    # plt.plot(np.arange(nr_sim_steps), log_comp_t, label=str(nx))

plt.plot(np.arange(nr_sim_steps)*dt, reference[0,:nr_sim_steps], 'k--', label='reference')
# plt.plot(np.arange(nr_sim_steps)*dt, np.ones(nr_sim_steps)*qlim, 'r-.', label='max')
# plt.plot(np.arange(nr_sim_steps)*dt, np.ones(nr_sim_steps)*-qlim, 'r-.', label='min')
plt.legend()
plt.grid()
plt.show()
