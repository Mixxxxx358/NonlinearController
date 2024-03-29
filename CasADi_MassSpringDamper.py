import numpy as np
import matplotlib.pyplot as plt
import time
import deepSI
from NonlinearController.controllers import VelocityMpcController
from NonlinearController.systems import FullMassSpringDamper
from NonlinearController.models import CasADi_model, odeCasADiMassSpringDamper


##################  System  #######################
dt = 0.01
k=1e4; c=5; m=0.25
system = FullMassSpringDamper(k, c, m, dt=dt, sigma_n=[1e-4])
system.reset_state()
nu = system.nu if system.nu is not None else 1
ny = system.ny if system.ny is not None else 1

##################  Model  #######################
ode = odeCasADiMassSpringDamper()
model = CasADi_model(ode, (1), dt, nx=2, nu=nu)
if model.expr_rk4 is not None:
    print("CasADi model")
else:
    print("ANN model")
    
##################  MPC controller  #######################
Nc=10; max_iter = 1; nr_sim_steps = 60
wlim = 70
qlim = 1e-2
nx = 2; nz = nx+ny; ne = 1


Q1 = np.zeros((ny,ny)); np.fill_diagonal(Q1, [5e5])
Q2 = np.zeros((nz,nz)); Q2[ny:,ny:] = np.eye(nx)
R = np.eye(nu)*1e-3
P = np.eye(ny)*0.1

##################  Initial Conditions  #######################
w0 = 0; q0 = [0.0]

##################  Reference  #######################
# a = 3; reference_x = np.hstack((np.ones(20)*a,np.ones(20)*-a,np.ones(60)*a))*1e-3
reference_x = deepSI.deepSI.exp_design.multisine(nr_sim_steps+Nc, pmax=20, n_crest_factor_optim=10)*4e-3
# reference_x = np.sin(np.arange(0,nr_sim_steps+Nc)/np.pi*2.5)*6e-3
reference = reference_x[np.newaxis]

##################  Control Loop 1  #######################
system.reset_state()
log_q_1 = np.zeros((ny,nr_sim_steps))
log_w_1 = np.zeros((nu,nr_sim_steps))

controller_1 = VelocityMpcController(system, model, Nc, Q1, Q2, R, P, qlim, wlim, nr_sim_steps=nr_sim_steps, \
                                     max_iter=max_iter, n_stages=1, numerical_method=4, model_simulation="LPV")

sim_start_time = time.time()

for k in range(nr_sim_steps):
    w0 = controller_1.QP_solve(reference_x[k])
    system.x = system.f(system.x, w0[0])
    dx_out, x_out = system.h(system.x, w0[0])
    q1 = x_out; x1 = np.vstack((dx_out, x_out))
    controller_1.update(q1, w0, x1)

    log_q_1[:,k] = q1
    log_w_1[:,k] = w0

sim_end_time = time.time()
print("Sim duration 1: " + str(sim_end_time - sim_start_time))

##################  Control Loop 2  #######################
system.reset_state()
log_q_2 = np.zeros((ny,nr_sim_steps))
log_w_2 = np.zeros((nu,nr_sim_steps))

controller_2 = VelocityMpcController(system, model, Nc, Q1, Q2, R, P, qlim, wlim, nr_sim_steps=nr_sim_steps, \
                                     max_iter=max_iter, n_stages=1, numerical_method=1, model_simulation="LPV")

sim_start_time = time.time()

for k in range(nr_sim_steps):
    w0 = controller_2.QP_solve(reference_x[k])
    system.x = system.f(system.x, w0[0])
    dx_out, x_out = system.h(system.x, w0[0])
    q1 = x_out; x1 = np.vstack((dx_out, x_out))
    controller_2.update(q1, w0, x1)

    log_q_2[:,k] = q1
    log_w_2[:,k] = w0

sim_end_time = time.time()
print("Sim duration 2: " + str(sim_end_time - sim_start_time))

##################  Control Loop 3  #######################
system.reset_state()
log_q_3 = np.zeros((ny,nr_sim_steps))
log_w_3 = np.zeros((nu,nr_sim_steps))

controller_3 = VelocityMpcController(system, model, Nc, Q1, Q2, R, P, qlim, wlim, nr_sim_steps=nr_sim_steps, \
                                     max_iter=max_iter, n_stages=1, numerical_method=3, model_simulation="LPV")

sim_start_time = time.time()

for k in range(nr_sim_steps):
    w0 = controller_3.QP_solve(reference_x[k])
    system.x = system.f(system.x, w0[0])
    dx_out, x_out = system.h(system.x, w0[0])
    q1 = x_out; x1 = np.vstack((dx_out, x_out))
    controller_3.update(q1, w0, x1)

    log_q_3[:,k] = q1
    log_w_3[:,k] = w0

sim_end_time = time.time()
print("Sim duration 3: " + str(sim_end_time - sim_start_time))

##################  Plots  #######################
fig1 = plt.figure(figsize=[8.9, 8])

plt.subplot(2,1,1)
plt.plot(np.arange(nr_sim_steps)*dt, log_w_1[0,:], label='MVT')
plt.plot(np.arange(nr_sim_steps)*dt, log_w_2[0,:], label='Rect')
plt.plot(np.arange(nr_sim_steps)*dt, log_w_3[0,:], label='Simps')
plt.plot(np.arange(nr_sim_steps)*dt, np.ones(nr_sim_steps)*wlim, 'r-.', label='max')
plt.plot(np.arange(nr_sim_steps)*dt, np.ones(nr_sim_steps)*-wlim, 'r-.', label='min')
plt.xlabel("time [s]")
plt.ylabel("voltage [V]")
plt.grid()
plt.legend(loc='lower right')


plt.subplot(2,1,2)
plt.plot(np.arange(nr_sim_steps)*dt, log_q_1[0,:], label='MVT')
plt.plot(np.arange(nr_sim_steps)*dt, log_q_2[0,:], label='Rect')
plt.plot(np.arange(nr_sim_steps)*dt, log_q_3[0,:], label='Simps')
plt.plot(np.arange(nr_sim_steps)*dt, reference[0,:nr_sim_steps], 'k--', label='reference')
plt.xlabel("time [s]")
plt.ylabel("angle theta")
plt.grid()
plt.legend(loc='lower right')

plt.show()