import numpy as np
import matplotlib.pyplot as plt
import time
import deepSI
from NonlinearController.utils import randomLevelReferenceSteps, wrapDisc, randomLevelReference, rmse
from NonlinearController.controllers import VelocityMpcController
from NonlinearController.systems import FullUnbalancedDisc
from NonlinearController.models import CasADi_model, odeCasADiUnbalancedDisc

##################  Simulation function  #######################
def controlLoop(reference_theta, system, model, nr_sim_steps, nu, ny, Q1, Q2, R, P, qlim, wlim, max_iter, n_stages, numerical_method, model_simulation):
    system.reset_state()
    log_q = np.zeros((ny,nr_sim_steps))
    log_w = np.zeros((nu,nr_sim_steps))

    controller_1 = VelocityMpcController(system, model, Nc, Q1, Q2, R, P, qlim, wlim, nr_sim_steps=nr_sim_steps, \
                                        max_iter=max_iter, n_stages=n_stages, numerical_method=numerical_method, model_simulation=model_simulation)

    sim_start_time = time.time()

    for k in range(nr_sim_steps):
        w0 = controller_1.QP_solve(reference_theta[k:k+Nc])
        system.x = system.f(system.x, w0[0])
        omega1, theta1 = system.h(system.x, w0[0])
        q1 = theta1; x1 = np.vstack((omega1, theta1))
        controller_1.update(q1, w0, x1)

        log_q[:,k] = q1
        log_w[:,k] = w0

    sim_end_time = time.time()

    return log_w, log_q

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
Nc=5; max_iter = 1; nr_sim_steps = 100
wlim = 8
qlim = 1000
nx = 2; nz = nx+ny; ne = 1

Q1 = np.zeros((ny,ny)); np.fill_diagonal(Q1, [50])
Q2 = np.zeros((nz,nz)); Q2[ny:,ny:] = np.eye(nx)*1
R = np.eye(nu)*1
P = np.eye(ny)*0.1

##################  Reference  #######################
a = 3.1; reference_theta = np.hstack((np.ones(20)*a,np.ones(20)*-a,np.ones(20)*a/2,np.ones(20)*-a/2,np.ones(40)*0))
# reference_theta = randomLevelReference(nr_sim_steps+Nc,[10,15],[-3.1,3.1])
# reference_theta = np.load("NonlinearController/references/setPoints.npy")
# reference_theta = deepSI.deepSI.exp_design.multisine(nr_sim_steps+Nc, pmax=20, n_crest_factor_optim=10)*1.5
# reference_theta = np.sin(np.arange(0,nr_sim_steps+Nc)/np.pi*2.0)*3.1
reference = reference_theta[np.newaxis]

##################  Control Loop MVT  #######################
steps = np.arange(1,10,1)
log_w_inf, log_q_inf = controlLoop(reference_theta, system, model, nr_sim_steps, nu, ny, Q1, Q2, R, P, qlim, wlim, max_iter, 200, 3, "LPV")

diff_MMVT = np.zeros(len(steps))
for i in range(len(steps)):
    log_w, log_q = controlLoop(reference_theta, system, model, nr_sim_steps, nu, ny, Q1, Q2, R, P, qlim, wlim, max_iter, steps[i], 4, "LPV")
    # diff_MMVT[i] = np.sum(np.abs(log_q[0,:] - log_q_inf[0,:]))
    diff_MMVT[i] = rmse(log_q[0,:], log_q_inf[0,:])

diff_Mid = np.zeros(len(steps))
for i in range(len(steps)):
    log_w, log_q = controlLoop(reference_theta, system, model, nr_sim_steps, nu, ny, Q1, Q2, R, P, qlim, wlim, max_iter, steps[i], 1, "LPV")
    # diff_Mid[i] = np.sum(np.abs(log_q[0,:] - log_q_inf[0,:]))
    diff_Mid[i] = rmse(log_q[0,:], log_q_inf[0,:])
    

diff_Trap = np.zeros(len(steps))
for i in range(len(steps)):
    log_w, log_q = controlLoop(reference_theta, system, model, nr_sim_steps, nu, ny, Q1, Q2, R, P, qlim, wlim, max_iter, steps[i], 2, "LPV")
    # diff_Trap[i] = np.sum(np.abs(log_q[0,:] - log_q_inf[0,:]))
    diff_Trap[i] = rmse(log_q[0,:], log_q_inf[0,:])

diff_Simps = np.zeros(len(steps))
for i in range(len(steps)):
    log_w, log_q = controlLoop(reference_theta, system, model, nr_sim_steps, nu, ny, Q1, Q2, R, P, qlim, wlim, max_iter, steps[i], 3, "LPV")
    # diff_Simps[i] = np.sum(np.abs(log_q[0,:] - log_q_inf[0,:]))
    diff_Simps[i] = rmse(log_q[0,:], log_q_inf[0,:])


##################  Plots  #######################
fig1 = plt.figure(figsize=[8.9, 8])
plt.plot(diff_MMVT, label="MMVT")
plt.plot(diff_Mid, label="Rect")
plt.plot(diff_Trap, label="Trap")
plt.plot(diff_Simps, label="Simpsons")
plt.legend()
plt.grid()
plt.show()
