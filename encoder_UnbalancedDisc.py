import numpy as np
import deepSI
import matplotlib.pyplot as plt
import time
from NonlinearController.utils import *
from NonlinearController.controllers import VelocityMpcController
from NonlinearController.systems import FullUnbalancedDisc
from NonlinearController.models import CasADi_model, odeCasADiUnbalancedDisc
import warnings
warnings.filterwarnings("ignore")

##################  System  #######################
dt = 0.1
system = FullUnbalancedDisc(dt=dt, sigma_n=[0.0])
system.reset_state()
nu = system.nu if system.nu is not None else 1
ny = system.ny if system.ny is not None else 1

##################  Model  #######################
model = deepSI.load_system("NonlinearController/trained_models/unbalanced/ObserverUnbalancedDisk_dt01_nab_4_SNR_30_e250")
# model = deepSI.load_system("NonlinearController/trained_models/unbalanced/ud_test_4")


##################  MPC controller parameters  #######################
Nc=5; max_iter = 1; nr_sim_steps = 100
wlim = 4
qlim = 1.2
nx = 2; nz = nx+ny; ne = 1

Q1 = np.zeros((ny,ny)); np.fill_diagonal(Q1, [1e1])
Q2 = np.zeros((nz,nz)); Q2[ny:,ny:] = np.eye(nx)*1
R = np.eye(nu)*0.1
P = np.eye(ny)*0.0001

##################  Initial Conditions  #######################
w0 = 0; q0 = [0.0]

##################  Reference  #######################
# a = 0.9; reference_theta = np.hstack((np.ones(20)*a,np.ones(20)*-a,np.ones(20)*a/2,np.ones(20)*-a/2,np.ones(40)*0))
# reference_theta = np.load("NonlinearController/references/setPoints.npy")
reference_theta = randomLevelReference(nr_sim_steps+Nc,[10,15],[-1,1])
# reference_theta = np.sin(np.arange(0,nr_sim_steps+Nc)/np.pi*3.5)*1

# reference_theta = deepSI.deepSI.exp_design.multisine(nr_sim_steps+Nc+20, pmax=19, n_crest_factor_optim=10)*0.6
# np.save("references/multisine.npy", reference_theta)
reference_theta = np.load("references/multisine.npy")

reference = reference_theta[np.newaxis]

##################  Control Loop MVT  #######################
system.reset_state()
log_q_1 = np.zeros((ny,nr_sim_steps))
log_w_1 = np.zeros((nu,nr_sim_steps))

controller_1 = VelocityMpcController(system, model, Nc, Q1, Q2, R, P, qlim, wlim, nr_sim_steps=nr_sim_steps, \
                                     max_iter=max_iter, n_stages=1, numerical_method=4, model_simulation="LPV", factorization_method="encoder", dev = "cpu")

sim_start_time = time.time()

for k in range(nr_sim_steps):
    w0 = controller_1.QP_solve(reference_theta[k:k+Nc])
    system.x = system.f(system.x, w0[0])
    omega1, theta1 = system.h(system.x, w0[0])
    q1 = theta1; x1 = np.vstack((omega1, theta1))
    controller_1.update(q1, w0, x1)

    log_q_1[:,k] = q1
    log_w_1[:,k] = w0

sim_end_time = time.time()
print("Sim duration 1: " + str(sim_end_time - sim_start_time))
print("Time breakdown: " + str(controller_1.computationTimeLogging()))

##################  Control Loop FTC 1 step  #######################
system.reset_state()
log_q_2 = np.zeros((ny,nr_sim_steps))
log_w_2 = np.zeros((nu,nr_sim_steps))

controller_2 = VelocityMpcController(system, model, Nc, Q1, Q2, R, P, qlim, wlim, nr_sim_steps=nr_sim_steps, \
                                     max_iter=max_iter, n_stages=1, numerical_method=1, model_simulation="LPV", factorization_method="encoder", dev = "cpu")

sim_start_time = time.time()

for k in range(nr_sim_steps):
    w0 = controller_2.QP_solve(reference_theta[k:k+Nc])
    system.x = system.f(system.x, w0[0])
    omega1, theta1 = system.h(system.x, w0[0])
    q1 = theta1; x1 = np.vstack((omega1, theta1))
    controller_2.update(q1, w0, x1)

    log_q_2[:,k] = q1
    log_w_2[:,k] = w0

sim_end_time = time.time()
print("Sim duration 2: " + str(sim_end_time - sim_start_time))
print("Time breakdown: " + str(controller_2.computationTimeLogging()))

##################  Control Loop FTC 10 step  #######################
system.reset_state()
log_q_3 = np.zeros((ny,nr_sim_steps))
log_w_3 = np.zeros((nu,nr_sim_steps))

controller_3 = VelocityMpcController(system, model, Nc, Q1, Q2, R, P, qlim, wlim, nr_sim_steps=nr_sim_steps, \
                                      max_iter=max_iter, n_stages=1, numerical_method=2, model_simulation="LPV", factorization_method="encoder", dev = "cpu")

sim_start_time = time.time()

for k in range(nr_sim_steps):
    w0 = controller_3.QP_solve(reference_theta[k:k+Nc])
    system.x = system.f(system.x, w0[0])
    omega1, theta1 = system.h(system.x, w0[0])
    q1 = theta1; x1 = np.vstack((omega1, theta1))
    controller_3.update(q1, w0, x1)

    log_q_3[:,k] = q1
    log_w_3[:,k] = w0

sim_end_time = time.time()
print("Sim duration 3: " + str(sim_end_time - sim_start_time))
print("Time breakdown: " + str(controller_3.computationTimeLogging()))

##################  Control Loop FTC 10 step  #######################
system.reset_state()
log_q_4 = np.zeros((ny,nr_sim_steps))
log_w_4 = np.zeros((nu,nr_sim_steps))

controller_4 = VelocityMpcController(system, model, Nc, Q1, Q2, R, P, qlim, wlim, nr_sim_steps=nr_sim_steps, \
                                      max_iter=max_iter, n_stages=1, numerical_method=3, model_simulation="LPV", factorization_method="encoder", dev = "cpu")

sim_start_time = time.time()

for k in range(nr_sim_steps):
    w0 = controller_4.QP_solve(reference_theta[k:k+Nc])
    system.x = system.f(system.x, w0[0])
    omega1, theta1 = system.h(system.x, w0[0])
    q1 = theta1; x1 = np.vstack((omega1, theta1))
    controller_4.update(q1, w0, x1)

    log_q_4[:,k] = q1
    log_w_4[:,k] = w0

sim_end_time = time.time()
print("Sim duration 3: " + str(sim_end_time - sim_start_time))
print("Time breakdown: " + str(controller_4.computationTimeLogging()))

##################  Plots  #######################
# load other experiments for comparison
ud_sqp_levels_u = np.load("experiments/ud_sqp_encoder_levels_u.npy")
ud_sqp_levels_q = np.load("experiments/ud_sqp_encoder_levels_q.npy")

ud_nmpc_levels_u = np.load("experiments/ud_nmpc_encoder_levels_u.npy")
ud_nmpc_levels_q = np.load("experiments/ud_nmpc_encoder_levels_q.npy")

ud_sqp_sinus_u = np.load("experiments/ud_sqp_encoder_sinus_u.npy")
ud_sqp_sinus_q = np.load("experiments/ud_sqp_encoder_sinus_q.npy")

ud_nmpc_sinus_u = np.load("experiments/ud_nmpc_encoder_sinus_u.npy")
ud_nmpc_sinus_q = np.load("experiments/ud_nmpc_encoder_sinus_q.npy")

fig1 = plt.figure(figsize=[8.9, 8])

plt.subplot(2,1,1)
# plt.plot(np.arange(nr_sim_steps)*dt, log_w_1[0,:], label='MMVT')
# plt.plot(np.arange(nr_sim_steps)*dt, log_w_2[0,:], label='Midpoint')
# plt.plot(np.arange(nr_sim_steps)*dt, log_w_3[0,:], label='Trapezian')
plt.plot(np.arange(nr_sim_steps)*dt, log_w_4[0,:], label='Simpsons')
# plt.plot(np.arange(nr_sim_steps)*dt, ud_sqp_levels_u[:,0], label='SQP')
# plt.plot(np.arange(nr_sim_steps)*dt, ud_nmpc_levels_u[0,:], label='ipopt')
plt.plot(np.arange(nr_sim_steps)*dt, np.hstack((0,ud_sqp_sinus_u[:-1,0])), label='SQP')
plt.plot(np.arange(nr_sim_steps)*dt, ud_nmpc_sinus_u[0,:], label='ipopt')
plt.plot(np.arange(nr_sim_steps)*dt, np.ones(nr_sim_steps)*wlim, 'r-.', label='max')
plt.plot(np.arange(nr_sim_steps)*dt, np.ones(nr_sim_steps)*-wlim, 'r-.', label='min')
plt.xlabel("time [s]")
plt.ylabel("voltage [V]")
plt.grid()
plt.legend(loc='lower right')


plt.subplot(2,1,2)
# plt.plot(np.arange(nr_sim_steps)*dt, log_q_1[0,:], label='MMVT')
# plt.plot(np.arange(nr_sim_steps)*dt, log_q_2[0,:], label='Midpoint')
# plt.plot(np.arange(nr_sim_steps)*dt, log_q_3[0,:], label='Trapezian')
plt.plot(np.arange(nr_sim_steps)*dt, log_q_4[0,:], label='Simpsons')
# plt.plot(np.arange(nr_sim_steps)*dt, np.hstack((0,ud_sqp_levels_q[:-1])), label='SQP')
# plt.plot(np.arange(nr_sim_steps)*dt, ud_nmpc_levels_q[0,:], label='ipopt')
plt.plot(np.arange(nr_sim_steps)*dt, np.hstack((0,ud_sqp_sinus_q[:-1])), label='SQP')
plt.plot(np.arange(nr_sim_steps)*dt, ud_nmpc_sinus_q[0,:], label='ipopt')
plt.plot(np.arange(nr_sim_steps)*dt, np.ones(nr_sim_steps)*qlim, 'r-.', label='max')
plt.plot(np.arange(nr_sim_steps)*dt, np.ones(nr_sim_steps)*-qlim, 'r-.', label='min')
plt.plot(np.arange(nr_sim_steps)*dt, reference[0,:nr_sim_steps], 'k--', label='reference')
plt.ylabel("angle theta [rad]")
plt.grid()
plt.legend(loc='lower right')

# plt.savefig("Figures/encoder_model_levels_methods.svg")
# plt.savefig("Figures/encoder_model_levels_controllers.svg")
# plt.savefig("Figures/encoder_model_sinus_methods.svg")
plt.savefig("Figures/encoder_model_sinus_controllers.svg")

plt.show()