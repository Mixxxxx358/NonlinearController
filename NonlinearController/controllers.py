import numpy as np
from NonlinearController.mpc_utils import *
from systems import LTI
from matplotlib import pyplot as plt
from casadi import *
from NonlinearController.model_utils import *

class Controller:
    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError

class Controller_mpc(Controller):
    def __init__(self, Nc, nx, nu, ny, Q=None, R=None):
        super(Controller_mpc,self).__init__()
        self.Nc = Nc
        self.nx = nx
        self.nu = nu
        self.ny = ny

        self.Q = Q if Q is not None else np.eye(self.nx)
        self.R = R if R is not None else np.eye(self.nu)

    def reset(self):
        raise NotImplementedError

class Controller_lin_mpc(Controller_mpc):
    def __init__(self, Nc, nx, nu, ny, A, B, C=None, D=None, f=None, h=None, Q=None, R=None): # we assume here that we always have A and B
        super(Controller_lin_mpc,self).__init__(Nc, nx, nu, ny, Q, R)

        self.A = A
        self.B = B
        self.C = C if C is not None else np.zeros((ny,nx))
        self.D = D if D is not None else np.zeros((ny,nu))

        self.list_A = np.tile(self.A, (self.Nc, 1))
        self.list_B = np.tile(self.B, (self.Nc, 1))

    def reset(self):
        raise NotImplementedError

    def __call__(self, reference, current_state): # currently converges to zero
        if len(reference) != self.Nc:
            pass

        self.Phi = get_Phi(self.list_A, self.Nc, self.nx)
        self.Gamma = get_Gamma(self.list_A, self.list_B, self.Nc, self.nx, self.nu)
        self.Omega = get_Omega(self.Nc, self.Q)
        self.Psi = get_Psi(self.Nc, self.R)

        self.G = 2*(self.Psi + self.Gamma.T @ self.Omega @ self.Gamma)
        self.F = 2*(self.Gamma.T @ self.Omega @ self.Phi)

        Uk = -np.linalg.inv(self.G) @ self.F @ current_state

        return Uk[:self.nu]

class Controller_lpv_mpc(Controller_mpc):
    def __init__(self, Nc, nx, nu, ny, Q=None, R=None):
        super(Controller_lpv_mpc,self).__init__(Nc, nx, nu, ny, Q, R)

        self.init_QP()

    def init_QP(self):
        self.list_A = np.zeros([Nc*nx, nx])
        self.list_B = np.zeros([Nc*nx, nu])
        self.list_C = np.zeros([Nc*ny, nx])

        self.Psi = get_Psi(Nc, self.R)
        self.Omega = get_Omega(Nc, self.Q)

        return

    def reset(self):
        raise NotImplementedError

    def __call__(self, reference, current_state):
        A = np.ones((self.nx, self.nx))
        B = np.ones((self.nx, self.nu))
        u = np.zeros((self.nu, 1))
        return A@current_state + B@u

    def iterative_QP_solve(self):
        # take lpv, solve qp for u, use model for new states. Repeat until convergence
        raise NotImplementedError

    def lpv(self):
        # take state and input predictions and return lpv representation of A,B,C,D matrices
        # pA = self.Get_A(np.vstack(np.split(x,Nc)).T,u)
        # for i in range(Nc):
        #     self.list_A[(nx*i):(nx*i+nx),:] = pA[:,i*nx:(i+1)*nx]

        # pB = self.Get_B(np.vstack(np.split(x,Nc)).T,u)
        # for i in range(Nc):
        #     self.list_B[(nx*i):(nx*i+nx),:] = pB[:,i*nu:(i+1)*nu]

        # pC = self.Get_C(np.vstack(np.split(x,Nc)).T,u)
        # for i in range(Nc):
        #     self.list_C[(ny*i):(ny*i+ny),:] = pC[:,i*nx:(i+1)*nx]
        raise NotImplementedError

class Controller_automatic_lpv_mpc(Controller_lpv_mpc):
    def __init__(self, Nc, nx, nu, ny, ss_enc, stages=5):
        super(Controller_automatic_lpv_mpc).__init__(Nc, nx, nu, ny)

        self.ss_enc
        self.stages = stages
        
    
    def init_automatic_lpv(self):
        x = MX.sym("x", self.nx, 1)
        u = MX.sym("u", self.nu, 1)

        rhs_f = CasADi_Fn(self.ss_enc, x, u)
        f = Function('f', [x, u], [rhs_f])
        rhs_y = CasADi_Hn(self.ss_enc, x)
        h = Function('h', [x], [rhs_y])

        self.correction_f = f(np.zeros((nx,1)), 0)
        rhs_fc = rhs_f - self.correction_f
        self.correction_h = h(np.zeros((nx,1)))
        rhs_yc = rhs_y - self.correction_h

        Jfx = Function("Jfx", [x, u], [jacobian(rhs_fc,x)])
        Jfu = Function("Jfu", [x, u], [jacobian(rhs_fc,u)])
        Jhx = Function("Jhx", [x, u], [jacobian(rhs_yc,x)])

        [A_sym, B_sym, C_sym] = self.lambda_simpson(x,u,nx,nu,ny,Jfx,Jfu,Jhx,self.stages)
        get_A = Function("get_A",[x,u],[A_sym])
        get_B = Function("get_B",[x,u],[B_sym])
        get_C = Function("get_C",[x,u],[C_sym])
        self.Get_A = get_A.map(self.Nc, "thread", 32)
        self.Get_B = get_B.map(self.Nc, "thread", 32)
        self.Get_C = get_C.map(self.Nc, "thread", 32)

        return

    def lambda_simpson(x,u,nx,nu,ny,Jfx,Jfu,Jhx,stages):
        # FUNCTION LAMBDA_SIMPSON
        # Simpson rule integrator between 0 and 1 with chosen resolution (stages)
        # used to get A,B matrices symbolically to be used at gridpoints
        
        A = np.zeros([nx,nx])
        B = np.zeros([nx,nu])
        C = np.zeros([ny,nx])
        lambda0 = 0
        dlam = 1/stages

        for i in np.arange(stages):
            A = A + dlam*1/6*(Jfx(lambda0*x,lambda0*u) + 4*Jfx((lambda0+dlam/2)*x,(lambda0+dlam/2)*u) + Jfx((lambda0+dlam)*x,(lambda0+dlam)*u))
            B = B + dlam*1/6*(Jfu(lambda0*x,lambda0*u) + 4*Jfu((lambda0+dlam/2)*x,(lambda0+dlam/2)*u) + Jfu((lambda0+dlam)*x,(lambda0+dlam)*u))
            C = C + dlam*1/6*(Jhx(lambda0*x,lambda0*u) + 4*Jhx((lambda0+dlam/2)*x,(lambda0+dlam/2)*u) + Jhx((lambda0+dlam)*x,(lambda0+dlam)*u))
            lambda0 = lambda0 + dlam
                
        return A,B,C

    def lpv(self):
        # take state and input predictions and return lpv representation of A,B,C,D matrices
        # self.list_A = 
        # self.list_B = 
        # self.list_C = 
        return

if __name__=='__main__':
    A = np.array([[-1/2,1/2],[1/2,0]]); B = np.array([[1],[0]])
    C = np.array([[1,0]])
    Nc = 5
    nx = 2; nu = 1; ny=1
    controller = Controller_lin_mpc(Nc,nx,nu,ny,A,B,C=C)
    system = LTI()
    system.reset_state()

    current_state = 0.2*np.ones((nx, 1))
    system.x = current_state[:,0]
    reference = np.zeros((ny, 1))

    Nsim = 5
    state_log = np.zeros((Nsim,nx))
    state_log[0] = current_state[:,0]

    for t in range(Nsim,1):
        u = controller(reference,current_state)
        system.x = system.f(system.x, u[:,0])
        current_state[:,:] = np.array([system.x]).T
        state_log[t] = current_state[:,0]

    plt.subplot(1,2,1)
    plt.plot(state_log[:,0])
    plt.subplot(1,2,2)
    plt.plot(state_log[:,1])
    plt.show()