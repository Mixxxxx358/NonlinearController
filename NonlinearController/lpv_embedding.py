import deepSI
import numpy as np
from casadi import *
from model_utils import *
from matplotlib import pyplot as plt

def lambda_trap(x0,dx,u0,du,nx,nu,ny,Jfx,Jfu,Jhx,stages):
    # FUNCTION LAMBDA_SIMPSON
    # Simpson rule integrator between 0 and 1 with chosen resolution (stages)
    # used to get A,B matrices symbolically to be used at gridpoints
    
    A = np.zeros([nx,nx])
    B = np.zeros([nx,nu])
    C = np.zeros([ny,nx])
    lambda0 = 0
    dlam = 1/stages

    x = lambda lam: x0 + lam*dx
    u = lambda lam: u0 + lam*du

    for i in np.arange(stages):
        A = A + dlam*1/2*(Jfx(x(lambda0), u(lambda0)) + Jfx(x(lambda0+dlam), u(lambda0+dlam)))
        B = B + dlam*1/2*(Jfu(x(lambda0), u(lambda0)) + Jfu(x(lambda0+dlam), u(lambda0+dlam)))
        C = C + dlam*1/2*(Jhx(x(lambda0), u(lambda0)) + Jhx(x(lambda0+dlam), u(lambda0+dlam)))
        lambda0 = lambda0 + dlam
            
    return A,B,C

if __name__=='__main__':
    model = deepSI.load_system("NonlinearController/trained_models/unbalanced/ObserverUnbalancedDisk_dt01_nab_4_SNR_30_e250")

    # declared sym variables
    nx = model.nx
    n_states = nx
    x = MX.sym("x",nx,1)
    nu = model.nu if model.nu is not None else 1
    n_controls = nu
    u = MX.sym("u",nu,1)
    ny = model.ny if model.ny is not None else 1

    # convert torch nn to casadi function
    x_rhs = CasADi_Fn(model, x, u)
    f = Function('f', [x, u], [x_rhs])
    y_rhs = CasADi_Hn(model, x)
    h = Function('h', [x], [y_rhs])

    Jfx = Function("Jfx", [x, u], [jacobian(x_rhs,x)])
    Jfu = Function("Jfu", [x, u], [jacobian(x_rhs,u)])
    Jhx = Function("Jhx", [x, u], [jacobian(y_rhs,x)])

    dx = MX.sym("dx",nx,1)
    x0 = MX.sym("x0",nx,1)
    du = MX.sym("du",nu,1)
    u0 = MX.sym("u0",nu,1)

    n_stages = 20
    [A_sym, B_sym, C_sym] = lambda_trap(x0,dx,u0,du,nx,nu,ny,Jfx,Jfu,Jhx,n_stages)
    get_A = Function("get_A",[x0,dx,u0,du],[A_sym])
    get_B = Function("get_B",[x0,dx,u0,du],[B_sym])
    get_C = Function("get_C",[x0,dx,u0,du],[C_sym])

    A = np.zeros((nx,nx))
    B = np.zeros((nx,nu))
    C = np.zeros((ny,nx))

    x0 = np.zeros((nx,1))
    dx = np.zeros((nx,1))
    u0 = np.zeros((nu,1))
    du = np.zeros((nu,1))

    steps = 100
    log_x = np.zeros((nx,steps))

    for i in range(steps):

        A[:,:] = get_A(x0,dx,u0,du)
        B[:,:] = get_B(x0,dx,u0,du)
        C[:,:] = get_C(x0,dx,u0,du)

        x1 = A @ x0 + B @ u0

        dx = x1-x0
        x0 = x1

        u1 = np.random.normal(size=(nu,1))
        du = u1 - u0
        u0 = u1

        log_x[:,i] = x1[:,0]

    plt.plot(log_x[0,:])
    plt.plot(log_x[1,:])
    plt.show()
