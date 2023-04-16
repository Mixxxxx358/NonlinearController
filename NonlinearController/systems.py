import deepSI
import numpy as np

class LTI(deepSI.System_ss):
    def __init__(self):
        super(LTI, self).__init__(nx=2, nu=None, ny=None)
    def f(self,x,u): #state function
        x = -0.5*x[0] + 0.5*x[1] + u, 0.5*x[0]
        return x
    def h(self,x,u): #output functions
        return x[0]

class UnbalancedDisc(deepSI.System_deriv):
    def __init__(self, dt=0.025, sigma_n=[0]):
        super(UnbalancedDisc, self).__init__(nx=2, dt=dt)
        self.g = 9.80155078791343
        self.J = 0.000244210523960356
        self.Km = 10.5081817407479
        self.I = 0.0410772235841364
        self.M = 0.0761844495320390
        self.tau = 0.397973147009910
        self.sigma_n = sigma_n

    def deriv(self,x,u):
        z1,z2 = x
        dz1 = -self.M*self.g*self.I/self.J*np.sin(z2) - 1/self.tau*z1 + self.Km/self.tau*u
        dz2 = z1
        return [dz1,dz2]

    def h(self,x,u):
        p = x[1]
        return (p+np.pi)%(2*np.pi) - np.pi
        # return x[1] + np.random.normal(0, self.sigma_n[0])
    
    # def reset_state(self):
        # self.x = np.zeros((self.nx,) if isinstance(self.nx,int) else self.nx)
        # self.x = np.array([np.pi,0])
    
class SinCosUnbalancedDisc(UnbalancedDisc):
    def __init__(self, dt=0.025, sigma_n=[0]):
        super(SinCosUnbalancedDisc, self).__init__(dt=dt, sigma_n=sigma_n)

    def h(self,x,u):
        measurement = x[1] + np.random.normal(0, self.sigma_n[0])
        return np.sin(measurement), np.cos(measurement)