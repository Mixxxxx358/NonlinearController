import numpy as np
import matplotlib.pyplot as plt
import time
import deepSI
from NonlinearController.utils import randomLevelReferenceSteps, wrapDisc, randomLevelReference
from NonlinearController.controllers import VelocityMpcController
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

##################  MPC controller parameters  #######################
Nc=10; max_iter = 8; nr_sim_steps = 20
wlim = 8
qlim = 1000
nx = 2; nz = nx+ny; ne = 1

Q1 = np.zeros((ny,ny)); np.fill_diagonal(Q1, [1e1])
Q2 = np.zeros((nz,nz)); Q2[ny:,ny:] = np.eye(nx)*1
R = np.eye(nu)*1
P = np.eye(ny)*0

##################  Reference  #######################
reference_theta = np.ones(nr_sim_steps+Nc)*3.1
reference = reference_theta[np.newaxis]

##################  Control Loop MVT  #######################
system.reset_state()
log_q_1 = np.zeros((ny,nr_sim_steps))
log_w_1 = np.zeros((nu,nr_sim_steps))

controller_1 = VelocityMpcController(system, model, Nc, Q1, Q2, R, P, qlim, wlim, nr_sim_steps=nr_sim_steps, \
                                     max_iter=max_iter, n_stages=10, numerical_method=3, model_simulation="LPV")

for k in range(nr_sim_steps):
    w0 = controller_1.QP_solve(reference_theta[k:k+Nc])
    system.x = system.f(system.x, w0[0])
    omega1, theta1 = system.h(system.x, w0[0])
    q1 = theta1; x1 = np.vstack((omega1, theta1))
    controller_1.update(q1, w0, x1)

    log_q_1[:,k] = q1
    log_w_1[:,k] = w0

log_iter_y = controller_1.log_iter_y

##################  Plots  #######################
fig2 = plt.figure(figsize=[8.9, 8])
plt.plot(np.arange(nr_sim_steps)*dt, log_q_1[0,:], label='1')
plt.plot(np.arange(nr_sim_steps)*dt, reference[0,:nr_sim_steps], 'k--', label='reference')
plt.xlabel("time [s]")
plt.ylabel("angle theta [rad]")
plt.grid()
plt.legend(loc='lower right')
plt.show()

fig1 = plt.figure(figsize=[8.9, 8])
for j in range(12):
    plt.subplot(2,6,j+1)
    for i in range(max_iter):
        plt.plot(log_iter_y[j,i,:], label=str(i))
    plt.plot(reference_theta[:Nc], 'k-.', label='reference')
    plt.grid()
    plt.legend()
plt.show()