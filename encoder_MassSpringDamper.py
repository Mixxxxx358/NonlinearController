import numpy as np
import deepSI
import matplotlib.pyplot as plt
from NonlinearController.controllers import VelocityMpcController
from NonlinearController.systems import FullMassSpringDamper
from NonlinearController.models import CasADi_model, odeCasADiUnbalancedDisc

##################  System  #######################
dt = 0.002
k=1e4; c=5; m=0.25
system = FullMassSpringDamper(k, c, m, dt=dt, sigma_n=[0])#1e-5
system.reset_state()
nu = system.nu if system.nu is not None else 1
ny = system.ny if system.ny is not None else 1

##################  Model  #######################
model = deepSI.load_system("NonlinearController/trained_models/MSD/msd_test_8")

##################  MPC controller  #######################
Nc=20; max_iter = 1; nr_sim_steps = 60
wlim = 50
qlim = 4e-3
nx = 2; nz = nx+ny; ne = 1

Q1 = np.zeros((ny,ny)); np.fill_diagonal(Q1, [500])
Q2 = np.zeros((nz,nz)); Q2[ny:,ny:] = np.eye(nx)*1
R = np.eye(nu)*1
P = np.eye(ny)*0.01

controller = VelocityMpcController(system, model, Nc, Q1, Q2, R, P, qlim, wlim, max_iter, n_stages=1, numerical_method=1)

##################  Initial Conditions  #######################
w0 = 0; q0 = [0.0]

##################  Reference  #######################
a = 3; reference_x = np.hstack((np.ones(20)*a,np.ones(20)*-a,np.ones(60)*a))*1e-3
reference = reference_x[np.newaxis]

##################  Logging  #######################
log_q = np.zeros((ny,nr_sim_steps))
log_w = np.zeros((nu,nr_sim_steps))

##################  Control Loop  #######################
for k in range(nr_sim_steps):
    w0 = controller.QP_solve(reference_x[k])
    system.x = system.f(system.x, w0[0])
    omega1, theta1 = system.h(system.x, w0[0])
    q1 = theta1; x1 = np.vstack((omega1, theta1))
    controller.update(q1, w0, x1)

    log_q[:,k] = q1
    log_w[:,k] = w0

##################  Plots  #######################
fig1 = plt.figure(figsize=[10, 6])

plt.subplot(2,1,1)
plt.plot(np.arange(nr_sim_steps)*dt, log_w[0,:], label='system input')
plt.plot(np.arange(nr_sim_steps)*dt, np.ones(nr_sim_steps)*wlim, 'r-.')#, label='max')
plt.plot(np.arange(nr_sim_steps)*dt, np.ones(nr_sim_steps)*-wlim, 'r-.')#, label='min')
plt.xlabel("time [s]")
plt.ylabel("input [N]")
plt.grid()
plt.legend(loc='lower right')


plt.subplot(2,1,2)
plt.plot(np.arange(nr_sim_steps)*dt, log_q[0,:], label='system output')
plt.plot(np.arange(nr_sim_steps)*dt, reference[0,:nr_sim_steps], '--', label='reference')
plt.plot(np.arange(nr_sim_steps)*dt, np.ones(nr_sim_steps)*qlim, 'r-.')#, label='max')
plt.plot(np.arange(nr_sim_steps)*dt, np.ones(nr_sim_steps)*-qlim, 'r-.')#, label='min')
plt.xlabel("time [s]")
plt.ylabel("position x [m]")
plt.grid()
plt.legend(loc='lower right')

plt.show()