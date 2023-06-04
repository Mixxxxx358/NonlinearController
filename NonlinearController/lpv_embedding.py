import deepSI
import numpy as np
from casadi import *
from NonlinearController.model_utils import *
from matplotlib import pyplot as plt
from functorch import jacrev, jacfwd, vmap
import torch
import time

def velocity_lambda_trap(x0,dx,u0,du,nx,nu,ny,Jfx,Jfu,Jhx,stages):
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

    # x = lambda lam: -(dx - x0) + lam*dx
    # u = lambda lam: -(du - u0) + lam*du

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

    def __call__(self, X, U):
        X_1 = np.hstack(np.split(X[:-self.nx],self.Nc))
        dX0 = np.hstack(np.split(X[self.nx:] - X[:-self.nx],self.Nc))
        U_1 = np.hstack(np.split(U[:-self.nu],self.Nc))
        dU0 = np.hstack(np.split(U[self.nu:] - U[:-self.nu],self.Nc))
        
        pA = self.lpv_A_Nc(X_1, dX0, U_1, dU0)
        pB = self.lpv_B_Nc(X_1, dX0, U_1, dU0)
        pC = self.lpv_C_Nc(X_1, dX0, U_1, dU0)

        return self.reshapeEmbedding(pA, pB, pC)

    def reshapeEmbedding(self, pA,pB,pC):
        list_A = np.zeros([self.Nc*self.nx, self.nx])
        list_B = np.zeros([self.Nc*self.nx, self.nu])
        list_C = np.zeros([self.Nc*self.ny, self.nx])

        for i in range(self.Nc):
            list_A[(self.nx*i):(self.nx*i+self.nx),:] = pA[:,i*self.nx:(i+1)*self.nx]

        for i in range(self.Nc):
            list_B[(self.nx*i):(self.nx*i+self.nx),:] = pB[:,i*self.nu:(i+1)*self.nu]

        for i in range(self.Nc):
            list_C[(self.ny*i):(self.ny*i+self.ny),:] = pC[:,i*self.nx:(i+1)*self.nx]

        return list_A, list_B, list_C
    
class velocity_lpv_embedder_autograd():
    def __init__(self, ss_enc, Nc, n_stages=20):
        if torch.cuda.is_available(): 
            dev = "cpu" 
        else: 
            dev = "cpu" 
        self.device = torch.device(dev) 
        print("Using " + str(dev))

        self.nx = ss_enc.nx
        self.nu = ss_enc.nu if ss_enc.nu is not None else 1
        self.ny = ss_enc.ny if ss_enc.ny is not None else 1

        self.Nc = Nc
        self.dlam = 1/n_stages

        self.JacF = torch.vmap(torch.func.jacrev(ss_enc.fn.to(self.device).float(), argnums=(0,1)))
        self.JacH = torch.vmap(torch.func.jacrev(ss_enc.hn.to(self.device).float()))

        self.mult_fA = (torch.from_numpy(np.tile(np.vstack((np.ones((1,self.nx,self.nx)), 4*np.ones((1,self.nx,self.nx)), np.ones((1,self.nx,self.nx)))),(n_stages*Nc,1,1)))*self.dlam/6).to(self.device)
        self.mult_fB = (torch.from_numpy(np.tile(np.vstack((np.ones((1,self.nx,self.nu)), 4*np.ones((1,self.nx,self.nu)), np.ones((1,self.nx,self.nu)))),(n_stages*Nc,1,1)))*self.dlam/6).to(self.device)
        self.mult_fC = (torch.from_numpy(np.tile(np.vstack((np.ones((1,self.ny,self.nx)), 4*np.ones((1,self.ny,self.nx)), np.ones((1,self.ny,self.nx)))),(n_stages*Nc,1,1)))*self.dlam/6).to(self.device)

        self.Lambda = np.array([])
        lambda0 = 0
        for i in np.arange(n_stages):
            self.Lambda = np.hstack((self.Lambda, lambda0, lambda0 + self.dlam/2, lambda0 + self.dlam)) # Simpson
            lambda0 = lambda0 + self.dlam
        n_int_comp = 3
        self.batch_size = Nc*n_stages*n_int_comp

    def __call__(self, X, U):
        X_1 = np.hstack(np.split(X[:-self.nx],self.Nc))
        dX0 = np.hstack(np.split(X[self.nx:] - X[:-self.nx],self.Nc))
        U_1 = np.hstack(np.split(U[:-self.nu],self.Nc))
        dU0 = np.hstack(np.split(U[self.nu:] - U[:-self.nu],self.Nc))

        Xlam = np.kron(dX0, self.Lambda) + np.kron(X_1, np.ones(self.Lambda.shape))
        Ulam = np.kron(dU0, self.Lambda) + np.kron(U_1, np.ones(self.Lambda.shape))

        x_tens = torch.reshape(torch.tensor(Xlam[np.newaxis].T, device=self.device),(self.batch_size,1,2)).float()
        u_tens = torch.reshape(torch.tensor(Ulam[np.newaxis].T, device=self.device),(self.batch_size,1,1)).float()
        fA, fB = self.JacF(x_tens,u_tens)
        fC = self.JacH(x_tens)

        return self.reshapeEmbedding(fA, fB, fC)

    def reshapeEmbedding(self, fA,fB,fC):
        list_A = np.zeros([self.Nc*self.nx, self.nx])
        list_B = np.zeros([self.Nc*self.nx, self.nu])
        list_C = np.zeros([self.Nc*self.ny, self.nx])

        tempA = torch.tensor_split(torch.mul(fA.view((self.batch_size,self.nx,self.nx)), self.mult_fA).detach(), self.Nc)
        tempB = torch.tensor_split(torch.mul(fB.view((self.batch_size,self.nx,self.nu)), self.mult_fB).detach(), self.Nc)
        tempC = torch.tensor_split(torch.mul(fC.view((self.batch_size,self.ny,self.nx)), self.mult_fC).detach(), self.Nc)
        for i in range(self.Nc):
            list_A[self.nx*(i):self.nx*(i+1),:] = torch.sum(tempA[i], axis=0).cpu().numpy()
            list_B[self.nx*(i):self.nx*(i+1),:] = torch.sum(tempB[i], axis=0).cpu().numpy()
            list_C[self.ny*(i):self.ny*(i+1),:] = torch.sum(tempC[i], axis=0).cpu().numpy()

        return list_A, list_B, list_C