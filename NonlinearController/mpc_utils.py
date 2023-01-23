import numpy as np
import itertools

def get_Phi(list_A, Nc, nx):
    '''Assemble Phi from predicted list A. Return numpy matrix of shape (Nc*nx, nx)'''
    Phi = np.zeros([nx*Nc, nx])
    for i in range(Nc):
        temp = np.eye(nx)
        for j in range(i,-1,-1):
            temp = np.matmul(temp, list_A[(nx*j):(nx*j+nx),:])
        Phi[i*nx:(i+1)*nx, :] = temp
    return Phi

def get_Gamma(list_A, list_B, Nc, nx, nu):
    '''Assemble Gamma from predicted list A and B. Return numpy matrix of shape (Nc*nu, Nc*nx)'''
    Gamma = np.zeros([nx*Nc, nu*Nc])
    for i in range(Nc):
        for j in range(0,i+1):
            temp = np.eye(nx)
            for l in range(i-j,-1,-1):
                if l == 0:
                    temp = np.matmul(temp, list_B[(nx*j):(nx*j+nx),:])
                else:
                    temp = np.matmul(temp, list_A[(nx*l):(nx*l+nx),:])
            Gamma[i*nx:nx*(i+1),j*nu:(j+1)*nu] = temp
    return Gamma

def get_Psi(Nc, R):
    '''Return kronecker product of R and Identity of size Nc'''
    return np.kron(np.eye(Nc), R)

def get_Omega(Nc, Q):
    '''Return kronecker product of Q and Identity of size Nc'''
    return np.kron(np.eye(Nc), Q)