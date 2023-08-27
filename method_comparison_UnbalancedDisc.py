import numpy as np
import matplotlib.pyplot as plt
import time
import deepSI
from NonlinearController.utils import randomLevelReferenceSteps, wrapDisc, randomLevelReference, rmse
from NonlinearController.controllers import VelocityMpcController, controlLoop
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
Nc=5; max_iter = 1; nr_sim_steps = 100
wlim = 5
qlim = 1.0
nx = 2; nz = nx+ny; ne = 1

Q1 = np.zeros((ny,ny)); np.fill_diagonal(Q1, [1e2])
Q2 = np.zeros((nz,nz)); Q2[ny:,ny:] = np.eye(nx)*1
R = np.eye(nu)*0.01
P = np.eye(ny)*0

##################  Reference  #######################
a = 3.1; reference_theta = np.hstack((np.ones(20)*a,np.ones(20)*-a,np.ones(20)*a/2,np.ones(20)*-a/2,np.ones(40)*0))
# reference_theta = np.load("NonlinearController/references/multisine.npy")

# reference_theta = randomLevelReference(nr_sim_steps+Nc,[10,15],[-3.1,3.1])
# reference_theta = np.load("NonlinearController/references/setPoints.npy")
# reference_theta = deepSI.deepSI.exp_design.multisine(nr_sim_steps+Nc, pmax=17, n_crest_factor_optim=10)*1.7
# reference_theta = np.sin(np.arange(0,nr_sim_steps+Nc)/np.pi*2.0)*3.1
reference = reference_theta[np.newaxis]

##################  Control Loop MVT  #######################
steps = np.arange(1,10,1)
log_w_inf, log_q_inf = controlLoop(reference_theta, system, model, Nc, nr_sim_steps, nu, ny, Q1, Q2, R, P, qlim, wlim, max_iter, 200, 3, "LPV")
inf_diff = (log_q_inf[0,:] - reference[0,:nr_sim_steps]); DR_inf = inf_diff.T @ inf_diff

diff_MMVT = np.zeros(len(steps))
for i in range(len(steps)):
    log_w, log_q = controlLoop(reference_theta, system, model, Nc, nr_sim_steps, nu, ny, Q1, Q2, R, P, qlim, wlim, max_iter, steps[i], 4, "LPV")
    # diff_MMVT[i] = np.sum(np.abs(log_q[0,:] - log_q_inf[0,:]))
    # diff_MMVT[i] = rmse(log_q[0,:], log_q_inf[0,:])
    MMVT_diff = (log_q[0,:] - reference[0,:nr_sim_steps]); DR_MMVT = MMVT_diff.T @ MMVT_diff
    diff_MMVT[i] = (DR_MMVT - DR_inf)/DR_inf

diff_Mid = np.zeros(len(steps))
for i in range(len(steps)):
    log_w, log_q = controlLoop(reference_theta, system, model, Nc, nr_sim_steps, nu, ny, Q1, Q2, R, P, qlim, wlim, max_iter, steps[i], 1, "LPV")
    # diff_Mid[i] = np.sum(np.abs(log_q[0,:] - log_q_inf[0,:]))
    # diff_Mid[i] = rmse(log_q[0,:], log_q_inf[0,:])
    MMVT_diff = (log_q[0,:] - reference[0,:nr_sim_steps]); DR_MMVT = MMVT_diff.T @ MMVT_diff
    diff_Mid[i] = (DR_MMVT - DR_inf)/DR_inf
    
diff_Trap = np.zeros(len(steps))
for i in range(len(steps)):
    log_w, log_q = controlLoop(reference_theta, system, model, Nc, nr_sim_steps, nu, ny, Q1, Q2, R, P, qlim, wlim, max_iter, steps[i], 2, "LPV")
    # diff_Trap[i] = np.sum(np.abs(log_q[0,:] - log_q_inf[0,:]))
    # diff_Trap[i] = rmse(log_q[0,:], log_q_inf[0,:])
    MMVT_diff = (log_q[0,:] - reference[0,:nr_sim_steps]); DR_MMVT = MMVT_diff.T @ MMVT_diff
    diff_Trap[i] = (DR_MMVT - DR_inf)/DR_inf

diff_Simps = np.zeros(len(steps))
for i in range(len(steps)):
    log_w, log_q = controlLoop(reference_theta, system, model, Nc, nr_sim_steps, nu, ny, Q1, Q2, R, P, qlim, wlim, max_iter, steps[i], 3, "LPV")
    # diff_Simps[i] = np.sum(np.abs(log_q[0,:] - log_q_inf[0,:]))
    # diff_Simps[i] = rmse(log_q[0,:], log_q_inf[0,:])
    MMVT_diff = (log_q[0,:] - reference[0,:nr_sim_steps]); DR_MMVT = MMVT_diff.T @ MMVT_diff
    diff_Simps[i] = (DR_MMVT - DR_inf)/DR_inf


##################  Plots  #######################
scaling = 6/10
fig1 = plt.figure(figsize=[8.9*scaling, 8*scaling])
plt.plot(diff_MMVT, label="MMVT")
plt.plot(diff_Mid, label="Midpoint")
plt.plot(diff_Trap, label="Trapezoidal")
plt.plot(diff_Simps, label="Simpsons")
plt.xlabel("nr integration steps")
plt.ylabel("RCSO")
plt.legend()
plt.grid()

plt.savefig("Figures/NumericalConvergenceLevels.svg")

plt.show()
