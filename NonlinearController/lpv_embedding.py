import deepSI
import numpy as np
from casadi import *
from NonlinearController.model_utils import *
from matplotlib import pyplot as plt

def velocity_lambda_trap(x0,dx,u0,du,nx,nu,ny,Jfx,Jfu,Jhx,stages):
    # FUNCTION LAMBDA_SIMPSON
    # Simpson rule integrator between 0 and 1 with chosen resolution (stages)
    # used to get A,B matrices symbolically to be used at gridpoints
    
    A = np.zeros([nx,nx])
    B = np.zeros([nx,nu])
    C = np.zeros([ny,nx])
    lambda0 = 0
    dlam = 1/stages

    x = lambda lam: -(dx - x0) + lam*dx
    u = lambda lam: -(du - u0) + lam*du

    for i in np.arange(stages):
        A = A + dlam*1/2*(Jfx(x(lambda0), u(lambda0)) + Jfx(x(lambda0+dlam), u(lambda0+dlam)))
        B = B + dlam*1/2*(Jfu(x(lambda0), u(lambda0)) + Jfu(x(lambda0+dlam), u(lambda0+dlam)))
        C = C + dlam*1/2*(Jhx(x(lambda0), u(lambda0)) + Jhx(x(lambda0+dlam), u(lambda0+dlam)))
        lambda0 = lambda0 + dlam
            
    return A,B,C

def lambda_trap(x0,u0,nx,nu,ny,Jfx,Jfu,Jhx,stages):
    # FUNCTION LAMBDA_SIMPSON
    # Simpson rule integrator between 0 and 1 with chosen resolution (stages)
    # used to get A,B matrices symbolically to be used at gridpoints
    
    A = np.zeros([nx,nx])
    B = np.zeros([nx,nu])
    C = np.zeros([ny,nx])
    lambda0 = 0
    dlam = 1/stages

    x = lambda lam: lam*x0
    u = lambda lam: lam*u0

    for i in np.arange(stages):
        A = A + dlam*1/2*(Jfx(x(lambda0), u(lambda0)) + Jfx(x(lambda0+dlam), u(lambda0+dlam)))
        B = B + dlam*1/2*(Jfu(x(lambda0), u(lambda0)) + Jfu(x(lambda0+dlam), u(lambda0+dlam)))
        C = C + dlam*1/2*(Jhx(x(lambda0), u(lambda0)) + Jhx(x(lambda0+dlam), u(lambda0+dlam)))
        lambda0 = lambda0 + dlam
            
    return A,B,C

def lpv_embedding(model, n_stages=20):
    '''Takes veocity ss encoder model and outputs velocity lpv embedded CasADi functions for A,B,C'''
    # declared sym variables
    nx = model.nx
    x = MX.sym("x",nx,1)
    nu = model.nu if model.nu is not None else 1
    u = MX.sym("u",nu,1)
    ny = model.ny if model.ny is not None else 1

    # convert torch nn to casadi function
    x_rhs = CasADi_Fn(model, x, u)
    f = Function('f', [x, u], [x_rhs])
    y_rhs = CasADi_Hn(model, x)
    h = Function('h', [x], [y_rhs])

    correction_f = f(np.zeros((nx,1)), 0)
    x_rhs_c = x_rhs - correction_f
    correction_h = h(np.zeros((nx,1)))
    y_rhs_c = y_rhs - correction_h

    Jfx = Function("Jfx", [x, u], [jacobian(x_rhs_c,x)])
    Jfu = Function("Jfu", [x, u], [jacobian(x_rhs_c,u)])
    Jhx = Function("Jhx", [x, u], [jacobian(y_rhs_c,x)])

    x0 = MX.sym("x0",nx,1)
    u0 = MX.sym("u0",nu,1)

    [A_sym, B_sym, C_sym] = lambda_trap(x0,u0,nx,nu,ny,Jfx,Jfu,Jhx,n_stages)
    lpv_A = Function("get_A",[x0,u0],[A_sym])
    lpv_B = Function("get_B",[x0,u0],[B_sym])
    lpv_C = Function("get_C",[x0,u0],[C_sym])

    return lpv_A, lpv_B, lpv_C, correction_f, correction_h

def velocity_lpv_embedding(model, n_stages=20):
    '''Takes veocity ss encoder model and outputs velocity lpv embedded CasADi functions for A,B,C'''
    # declared sym variables
    nx = model.nx
    x = MX.sym("x",nx,1)
    nu = model.nu if model.nu is not None else 1
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

    [A_sym, B_sym, C_sym] = velocity_lambda_trap(x0,dx,u0,du,nx,nu,ny,Jfx,Jfu,Jhx,n_stages)
    lpv_A = Function("get_A",[x0,dx,u0,du],[A_sym])
    lpv_B = Function("get_B",[x0,dx,u0,du],[B_sym])
    lpv_C = Function("get_C",[x0,dx,u0,du],[C_sym])

    return lpv_A, lpv_B, lpv_C

class velocity_lpv_embedder():
    def __init__(self, ss_enc, Nc, n_stages=20):
        self.nx = ss_enc.nx
        self.nu = ss_enc.nu if ss_enc.nu is not None else 1
        self.ny = ss_enc.ny if ss_enc.ny is not None else 1

        self.Nc = Nc

        self.lpv_A, self.lpv_B, self.lpv_C = velocity_lpv_embedding(ss_enc,n_stages=n_stages)
        self.lpv_A_Nc = self.lpv_A.map(self.Nc, "thread", 32)
        self.lpv_B_Nc = self.lpv_B.map(self.Nc, "thread", 32)
        self.lpv_C_Nc = self.lpv_C.map(self.Nc, "thread", 32)

    def __call__(self, dX, x_1, dU, u_1):
        list_A = np.zeros([self.Nc*self.nx, self.nx])
        list_B = np.zeros([self.Nc*self.nx, self.nu])
        list_C = np.zeros([self.Nc*self.ny, self.nx])

        X_1 = np.zeros((self.nx,self.Nc))
        X_1[:,0] = x_1

        for i in range(self.Nc):
            X_1[:,i] = dX[i,:]

        return list_A, list_B, list_C