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
Nc=5; max_iter = 1; nr_sim_steps = 100
wlim = 5
qlim = 1000
nx = 2; nz = nx+ny; ne = 1

Q1 = np.zeros((ny,ny)); np.fill_diagonal(Q1, [1e2])
Q2 = np.zeros((nz,nz)); Q2[ny:,ny:] = np.eye(nx)*1
R = np.eye(nu)*0.01
P = np.eye(ny)*0

##################  Reference  #######################
a = 3.1; reference_theta = np.hstack((np.ones(20)*a,np.ones(20)*-a,np.ones(20)*a/2,np.ones(20)*-a/2,np.ones(40)*0))
# a=3.1; reference_theta = []
# for i in range(20):
#     reference_theta = np.hstack((reference_theta, np.ones(5)*a/20*i))
# reference_theta = np.hstack((reference_theta, np.ones(80)*a))
# reference_theta = np.load("references/multisine.npy")

# reference_theta = randomLevelReference(nr_sim_steps+Nc,[10,15],[-3.1,3.1])
# reference_theta = np.load("NonlinearController/references/setPoints.npy")
# reference_theta = deepSI.deepSI.exp_design.multisine(nr_sim_steps+Nc+20, pmax=17, n_crest_factor_optim=10)*1.7
# np.save("NonlinearController/references/multisine.npy", reference_theta)
# reference_theta = np.sin(np.arange(0,nr_sim_steps+Nc)/np.pi*1.0)*3.1
reference = reference_theta[np.newaxis]

##################  Control Loop MVT  #######################
system.reset_state()
log_q_1 = np.zeros((ny,nr_sim_steps))
log_w_1 = np.zeros((nu,nr_sim_steps))

controller_1 = VelocityMpcController(system, model, Nc, Q1, Q2, R, P, qlim, wlim, nr_sim_steps=nr_sim_steps, \
                                     max_iter=1, n_stages=1, numerical_method=4, model_simulation="LPV")

sim_start_time = time.time()

for k in range(nr_sim_steps):
    w0 = controller_1.QP_solve(reference_theta[k:Nc+k])
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
                                     max_iter=1, n_stages=1, numerical_method=1, model_simulation="LPV")

sim_start_time = time.time()

for k in range(nr_sim_steps):
    w0 = controller_2.QP_solve(reference_theta[k:Nc+k])
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
                                     max_iter=max_iter, n_stages=1, numerical_method=2, model_simulation="LPV")

sim_start_time = time.time()

for k in range(nr_sim_steps):
    w0 = controller_3.QP_solve(reference_theta[k:Nc+k])
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
                                     max_iter=max_iter, n_stages=1, numerical_method=3, model_simulation="LPV")

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
print("Time breakdown: " + str(controller_3.computationTimeLogging()))

##################  Plots  #######################
# load other experiments for comparison
ud_sqp_levels_u = np.load("NonlinearController/experiments/ud_sqp_levels_u.npy")
ud_sqp_levels_q = np.load("NonlinearController/experiments/ud_sqp_levels_q.npy")

ud_nmpc_levels_u = np.load("NonlinearController/experiments/ud_nmpc_levels_u.npy")
ud_nmpc_levels_q = np.load("NonlinearController/experiments/ud_nmpc_levels_q.npy")

ud_sqp_sinus_u = np.load("NonlinearController/experiments/ud_sqp_sinus_u.npy")
ud_sqp_sinus_q = np.load("NonlinearController/experiments/ud_sqp_sinus_q.npy")

ud_nmpc_sinus_u = np.load("NonlinearController/experiments/ud_nmpc_sinus_u.npy")
ud_nmpc_sinus_q = np.load("NonlinearController/experiments/ud_nmpc_sinus_q.npy")


# fig1 = plt.figure(figsize=[8.9, 8])
scaling = 6/10
fig1 = plt.figure(figsize=[8.9*scaling, 7*scaling])

plt.subplot(2,1,1)
# plt.plot(np.arange(nr_sim_steps)*dt, log_w_1[0,:], label='MMVT')
# plt.plot(np.arange(nr_sim_steps)*dt, log_w_2[0,:], label='Midpoint')
# plt.plot(np.arange(nr_sim_steps)*dt, log_w_3[0,:], label='Trapezoidal')
plt.plot(np.arange(nr_sim_steps)*dt, log_w_4[0,:], label='Simpsons')
plt.plot(np.arange(nr_sim_steps)*dt, ud_sqp_levels_u[:,0], label='SQP')
plt.plot(np.arange(nr_sim_steps)*dt, ud_nmpc_levels_u[0,:], label='ipopt')
# plt.plot(np.arange(nr_sim_steps)*dt, np.hstack((ud_sqp_sinus_u[:,0])), label='SQP')
# plt.plot(np.arange(nr_sim_steps)*dt, ud_nmpc_sinus_u[0,:], label='ipopt')
plt.plot(np.arange(nr_sim_steps)*dt, np.ones(nr_sim_steps)*wlim, 'r-.', label='max')
plt.plot(np.arange(nr_sim_steps)*dt, np.ones(nr_sim_steps)*-wlim, 'r-.', label='min')
# plt.xlabel("time [s]")
plt.ylabel("voltage [V]")
plt.grid()
plt.legend(loc='lower right')

plt.subplot(2,1,2)
# plt.plot(np.arange(nr_sim_steps)*dt, log_q_1[0,:], label='MMVT')
# plt.plot(np.arange(nr_sim_steps)*dt, log_q_2[0,:], label='Midpoint')
# plt.plot(np.arange(nr_sim_steps)*dt, log_q_3[0,:], label='Trapezoidal')
plt.plot(np.arange(nr_sim_steps)*dt, log_q_4[0,:], label='Simpsons')
plt.plot(np.arange(nr_sim_steps)*dt, ud_sqp_levels_q[:-1,1], label='SQP')
plt.plot(np.arange(nr_sim_steps)*dt, ud_nmpc_levels_q[1,:], label='ipopt')
# plt.plot(np.arange(nr_sim_steps)*dt, ud_sqp_sinus_q[:-1,1], label='SQP')
# plt.plot(np.arange(nr_sim_steps)*dt, ud_nmpc_sinus_q[1,:], label='ipopt')
plt.plot(np.arange(nr_sim_steps)*dt, reference[0,:nr_sim_steps], 'k--', label='reference')
# plt.plot(np.arange(nr_sim_steps)*dt, np.ones(nr_sim_steps)*np.pi, 'r-.')#, label='max')
# plt.plot(np.arange(nr_sim_steps)*dt, np.ones(nr_sim_steps)*-np.pi, 'r-.')#, label='min')
plt.xlabel("time [s]")
plt.ylabel("angle theta [rad]")
plt.grid()
plt.legend(loc='lower right')

# plt.savefig("Figures/CasADi_model_levels_methods.svg")
plt.savefig("Figures/CasADi_model_levels_controllers.svg")
# plt.savefig("Figures/CasADi_model_sinus_methods.svg")
# plt.savefig("Figures/CasADi_model_sinus_controllers.svg")

plt.show()

# Q_DR = np.ones(1)
# ipopt_diff = (ud_nmpc_sinus_q[1,:] - reference[0,:nr_sim_steps]); DR_ipopt = ipopt_diff.T @ ipopt_diff
# sqp_diff = (ud_sqp_sinus_q[:-1,1] - reference[0,:nr_sim_steps]); DR_sqp = sqp_diff.T @ sqp_diff
# simpsons_diff = (log_q_3[0,:] - reference[0,:nr_sim_steps]); DR_simpsons = simpsons_diff.T @ simpsons_diff

# RCSO_sqp = (DR_sqp - DR_ipopt)/DR_ipopt
# RCSO_simpsons = (DR_simpsons - DR_ipopt)/DR_ipopt

# # print([DR_ipopt, DR_sqp, DR_simpsons])
# print([RCSO_sqp, RCSO_simpsons])
