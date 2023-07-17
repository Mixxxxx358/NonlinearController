import numpy as np
import random

def wrapDisc(list):
    return (list+np.pi)%(2*np.pi) - np.pi

def randomLevelReference(Nsim, nt_range, level_range):
    x_reference_list = np.array([])
    Nsim_remaining = Nsim
    while True:
        Nsim_steps = random.randint(nt_range[0],nt_range[1])
        Nsim_remaining = Nsim_remaining - Nsim_steps
        x_reference_list = np.hstack((x_reference_list, np.ones(Nsim_steps)*random.uniform(level_range[0],level_range[1])))

        if Nsim_remaining <= 0:
            x_reference_list = x_reference_list[:Nsim]
            break
    return x_reference_list

def randomLevelReferenceSteps(Nsim, nt_range, level_range):
    x_reference_list = np.array([])
    Nsim_remaining = Nsim
    current_amp = 0
    while True:
        Nsim_steps = random.randint(nt_range[0],nt_range[1])
        Nsim_remaining = Nsim_remaining - Nsim_steps
        current_amp += random.uniform(level_range[0],level_range[1])

        x_reference_list = np.hstack((x_reference_list, np.ones(Nsim_steps)*current_amp))

        if Nsim_remaining <= 0:
            x_reference_list = x_reference_list[:Nsim]
            break
    return x_reference_list

def setPointInput(y_ref):
    g = 9.80155078791343
    J = 0.000244210523960356
    Km = 10.5081817407479
    I = 0.0410772235841364
    M = 0.0761844495320390
    tau = 0.397973147009910

    return -(tau * M * g * I)/(Km * J) * np.sin(y_ref)

def my_rk4(x,u,f,dt,M):
    # function MY_RK4
    # used for a discretized simulation of next time-step
    # of the state variables using f(x,u)
    dt = dt/M
    x_next = x
    for i in np.arange(M):
        k1 = dt*f(x,u)
        k2 = dt*f(x+1*k1/2,u)
        k3 = dt*f(x+1*k2/2,u)
        k4 = dt*f(x+1*k3,u)
        x_next = x_next + 1/6*1*(k1+2*k2+2*k3+k4)
    
    return x_next

def differenceVector(X_1, nx):
    return X_1[nx:] - X_1[:-nx]

def extendABC(list_A, list_B, list_C, nx, ny, nu, Nc):
    nz = nx+ny
    list_ext_A = np.zeros((nz*Nc,nz))
    list_ext_B = np.zeros((nz*Nc,nu))
    list_ext_C = np.zeros((ny*Nc,nz))

    for i in range(Nc):
        list_ext_A[(i*nz):(i*nz+ny),:ny] = np.eye(ny)
        list_ext_A[(i*nz):(i*nz+ny),ny:nz] = list_C[(ny*i):(ny*i+ny),:]
        list_ext_A[(i*nz+ny):(i*nz+nz),ny:nz] = list_A[(nx*i):(nx*i+nx),:]

    for i in range(Nc):
        list_ext_B[(i*nz+ny):(i*nz+nz),:] = list_B[(nx*i):(nx*i+nx),:]

    for i in range(Nc):
        list_ext_C[(i*ny):(i*ny+ny),:ny] = np.eye(ny)
        list_ext_C[(i*ny):(i*ny+ny),ny:nz] = list_C[(ny*i):(ny*i+ny),:]
    
    return list_ext_A, list_ext_B, list_ext_C

def extendState(Y_1, dX0, nx, ny, Nc):
    nz = nx+ny
    Z0 = np.zeros((Nc*nz,1))

    for i in range(Nc):
        Z0[(nz*i):(nz*i+ny),:] = Y_1[ny*i:(ny*i+ny),:]
        Z0[(nz*i+ny):(nz*i+nz),:] = dX0[nx*i:(nx*i+nx),:]

    return Z0

def decodeState(Z0, nx, ny, Nc):
    nz = nx+ny
    Y_1 = np.zeros((Nc*ny,1))
    dX0 = np.zeros((Nc*nx,1))

    for i in range(Nc):
        Y_1[ny*i:(ny*i+ny),:] = Z0[(nz*i):(nz*i+ny),:]
        dX0[nx*i:(nx*i+nx),:] = Z0[(nz*i+ny):(nz*i+nz),:]

    return Y_1, dX0

def extendReference(reference, nx, ny, Nc):
    nz = nx+ny
    r = np.zeros((Nc*nz,1))

    for i in range(Nc):
        r[nz*i:nz*i+ny,0] = reference[:,i]

    return r

class normalizer():
    def __init__(self, y0, ystd, u0, ustd):
        self.y0 = y0
        self.ystd = ystd

        self.u0 = u0
        self.ustd = ustd