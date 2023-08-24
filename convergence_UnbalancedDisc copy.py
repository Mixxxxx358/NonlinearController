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
Nc= 10; nr_sim_steps = 30
wlim = 8
qlim = 100.0
nx = 2; nz = nx+ny; ne = 1

Q1 = np.zeros((ny,ny)); np.fill_diagonal(Q1, [1e3])
Q2 = np.zeros((nz,nz)); Q2[ny:,ny:] = np.eye(nx)*1
R = np.eye(nu)*0.1
P = np.eye(ny)*0

##################  Reference  #######################
reference_theta = np.hstack((np.zeros(10),np.ones(90)*2.7))
reference = reference_theta[np.newaxis]

##################  Start Loop  #######################
fig1 = plt.figure(figsize=[8.9, 8])
range_max_iter = 101
step_size = 20
for max_iter in range(1,range_max_iter,step_size):

    ##################  Control Loop MVT  #######################
    log_w, log_q, log_comp_t = controlLoopTime(reference_theta, system, model, Nc, nr_sim_steps, nu, ny, Q1, Q2, R, P, qlim, wlim, max_iter, 10, 2, "True", future_reference=False)

    ##################  Plots  #######################
    plt.plot(np.arange(nr_sim_steps)*dt, log_q[0,:], label=str(max_iter), color=[0.1, (1.0/range_max_iter)*max_iter, (1.0/range_max_iter)*max_iter])


##################  Plots  #######################
plt.plot(np.arange(nr_sim_steps)*dt, reference[0,:nr_sim_steps], 'r--', label='reference')
plt.xlabel("integration steps")
plt.legend()
plt.grid()
plt.show()
