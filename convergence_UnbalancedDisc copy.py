import numpy as np
import matplotlib.pyplot as plt
import time
import deepSI
from NonlinearController.utils import randomLevelReferenceSteps, wrapDisc, randomLevelReference, rmse
from NonlinearController.controllers import VelocityMpcController, controlLoopTime
from NonlinearController.systems import FullUnbalancedDisc
from NonlinearController.models import CasADi_model, odeCasADiUnbalancedDisc

##################  System  #######################
dt = 0.1
system = FullUnbalancedDisc(dt=dt, sigma_n=[0.0])
system.reset_state()
nu = system.nu if system.nu is not None else 1
ny = system.ny if system.ny is not None else 1

##################  Model  #######################
ode = odeCasADiUnbalancedDisc()
model = CasADi_model(ode, (1), dt, nx=2, nu=nu)
# model = deepSI.load_system("NonlinearController/trained_models/unbalanced/ud_test_4")

##################  MPC controller parameters  #######################
Nc= 5; nr_sim_steps = 20
wlim = 5
qlim = 100.0
nx = 2; nz = nx+ny; ne = 1

Q1 = np.zeros((ny,ny)); np.fill_diagonal(Q1, [1e2])
Q2 = np.zeros((nz,nz)); Q2[ny:,ny:] = np.eye(nx)*1
R = np.eye(nu)*0.01
P = np.eye(ny)*0

##################  Reference  #######################
reference_theta = np.hstack((np.zeros(10),np.ones(90)*2.8))
reference = reference_theta[np.newaxis]

##################  Start Loop  #######################
scaling = 6/10
fig1 = plt.figure(figsize=[8.9*scaling, 8*scaling])
range_max_iter = 6
step_size = 1
for max_iter in range(1,range_max_iter,step_size):

    ##################  Control Loop MVT  #######################
    log_w, log_q, log_comp_t = controlLoopTime(reference_theta, system, model, Nc, nr_sim_steps, nu, ny, Q1, Q2, R, P, qlim, wlim, max_iter, 1, 1, "LPV", future_reference=True)

    ##################  Plots  #######################
    plt.subplot(2,1,1)
    plt.plot(np.arange(nr_sim_steps)*dt, log_w[0,:], label=str(max_iter) + " iter")#, color=[0.1, (1.0/range_max_iter)*max_iter, (1.0/range_max_iter)*max_iter])
    plt.subplot(2,1,2)
    plt.plot(np.arange(nr_sim_steps)*dt, log_q[0,:], label=str(max_iter) + " iter")#, color=[0.1, (1.0/range_max_iter)*max_iter, (1.0/range_max_iter)*max_iter])


##################  Plots  #######################
plt.plot(np.arange(nr_sim_steps)*dt, reference[0,:nr_sim_steps], 'k--', label='reference')
plt.xlabel("time [s]")
plt.ylabel("angle theta [rad]")
plt.legend()
plt.grid()
plt.subplot(2,1,1)
plt.ylabel("voltage [V]")
plt.grid()
plt.legend()

plt.savefig("Figures/convergence_trajectories.svg")

plt.show()
