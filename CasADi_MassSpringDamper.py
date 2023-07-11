import numpy as np
import matplotlib.pyplot as plt
from NonlinearController.controllers import VelocityMpcController
from NonlinearController.systems import FullMassSpringDamper
from NonlinearController.models import CasADi_model, odeCasADiMassSpringDamper
import time

##################  System  #######################
dt = 0.002
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
Nc=5; max_iter = 5; nr_sim_steps = 60
wlim = 20
qlim = 1
nx = 2; nz = nx+ny; ne = 1


Q1 = np.zeros((ny,ny)); np.fill_diagonal(Q1, [5e5])
Q2 = np.zeros((nz,nz)); Q2[ny:,ny:] = np.eye(nx)
R = np.eye(nu)*1e-5
P = np.eye(ny)*0.1

controller = VelocityMpcController(system, model, Nc, Q1, Q2, R, P, qlim, wlim, max_iter, n_stages=1, numerical_method=1)

##################  Initial Conditions  #######################
w0 = 0; q0 = [0.0]

##################  Reference  #######################
a = 1.5; reference_x = np.hstack((np.ones(20)*a,np.ones(20)*0,np.ones(60)*-a))*1e-3
reference = reference_x[np.newaxis]

##################  Logging  #######################
log_q = np.zeros((ny,nr_sim_steps))
log_w = np.zeros((nu,nr_sim_steps))

##################  Control Loop  #######################
sim_start_time = time.time()

for k in range(nr_sim_steps):
    w0 = controller.QP_solve(reference_x[k])
    system.x = system.f(system.x, w0[0])
    dx_out, x_out = system.h(system.x, w0[0])
    q1 = x_out; x1 = np.vstack((dx_out, x_out))
    controller.update(q1, w0, x1)

    log_q[:,k] = q1
    log_w[:,k] = w0

sim_end_time = time.time()
print("Sim duration: " + str(sim_end_time - sim_start_time))

##################  Plots  #######################
fig1 = plt.figure(figsize=[10, 6])

plt.subplot(2,1,1)
plt.plot(np.arange(nr_sim_steps)*dt, log_w[0,:], label='system input')
# plt.plot(np.arange(nr_sim_steps)*dt, np.hstack((log_u_lpv[1:],log_u_lpv[-1])), label='lpv input')
# plt.plot(np.arange(nr_sim_steps)*dt, np.ones(nr_sim_steps)*wlim, 'r-.')#, label='max')
# plt.plot(np.arange(nr_sim_steps)*dt, np.ones(nr_sim_steps)*-wlim, 'r-.')#, label='min')
plt.xlabel("time [s]")
plt.ylabel("voltage [V]")
plt.grid()
plt.legend(loc='lower right')


plt.subplot(2,1,2)
plt.plot(np.arange(nr_sim_steps)*dt, log_q[0,:], label='system output')
plt.plot(np.arange(nr_sim_steps)*dt, reference[0,:nr_sim_steps], '--', label='reference')
plt.xlabel("time [s]")
plt.ylabel("angle theta")
plt.grid()
plt.legend(loc='lower right')

plt.show()