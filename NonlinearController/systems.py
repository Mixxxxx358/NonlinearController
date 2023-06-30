import deepSI
import numpy as np

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
        p = x[1] + np.random.normal(0, self.sigma_n[0])
        p = (p+np.pi)%(2*np.pi) - np.pi
        # return x[0], p
        return p
    
class FullUnbalancedDisc(UnbalancedDisc):
    def __init__(self, dt=0.025, sigma_n=[0]):
        super(FullUnbalancedDisc, self).__init__(dt=dt, sigma_n=sigma_n)

    def h(self,x,u):
        theta = x[1] + np.random.normal(0, self.sigma_n[0])
        # theta = (theta+np.pi)%(2*np.pi) - np.pi
        omega = x[0] + np.random.normal(0, self.sigma_n[0])
        return omega, theta
    
class SinCosUnbalancedDisc(UnbalancedDisc):
    def __init__(self, dt=0.025, sigma_n=[0]):
        super(SinCosUnbalancedDisc, self).__init__(dt=dt, sigma_n=sigma_n)

    def h(self,x,u):
        measurement = x[1] + np.random.normal(0, self.sigma_n[0])
        return np.sin(measurement), np.cos(measurement)
    
class ReversedUnbalancedDisc(deepSI.System_deriv):
    def __init__(self, dt=0.025, sigma_n=[0]):
        super(ReversedUnbalancedDisc, self).__init__(nx=2, dt=dt)
        self.g = 9.80155078791343
        self.J = 0.000244210523960356
        self.Km = 10.5081817407479
        self.I = 0.0410772235841364
        self.M = 0.0761844495320390
        self.tau = 0.397973147009910
        self.sigma_n = sigma_n

    def deriv(self,x,u):
        z1,z2 = x
        dz1 = self.M*self.g*self.I/self.J*np.sin(z2) - 1/self.tau*z1 + self.Km/self.tau*u
        dz2 = z1
        return [dz1,dz2]

    def h(self,x,u):
        p = x[1] + np.random.normal(0, self.sigma_n[0])
        p = (p+np.pi)%(2*np.pi) - np.pi + np.random.normal(0, self.sigma_n[0])
        return p

class ReversedFullUnbalancedDisc(ReversedUnbalancedDisc):
    def __init__(self, dt=0.025, sigma_n=[0]):
        super(ReversedFullUnbalancedDisc, self).__init__(dt=dt, sigma_n=sigma_n)

    def h(self,x,u):
        theta = x[1] + np.random.normal(0, self.sigma_n[0])
        # theta = (theta+np.pi)%(2*np.pi) - np.pi
        omega = x[0] + np.random.normal(0, self.sigma_n[0])
        return omega, theta
   
class ReversedSinCosUnbalancedDisc(ReversedUnbalancedDisc):
    def __init__(self, dt=0.025, sigma_n=[0]):
        super(ReversedSinCosUnbalancedDisc, self).__init__(dt=dt, sigma_n=sigma_n)

    def h(self,x,u):
        measurement = x[1] + np.random.normal(0, self.sigma_n[0])
        return np.sin(measurement), np.cos(measurement)

class MassSpringDamper(deepSI.System_deriv):
    def __init__(self, k, c, m, dt=0.025, sigma_n=[0]):
        super(MassSpringDamper, self).__init__(nx=2, dt=dt)
        self.k = k
        self.c = c
        self.m = m

    def deriv(self,x,u):
        z1,z2 = x
        dz1 = -self.k/self.m*z2 - self.c/self.m*z1 + 1/self.m*u
        dz2 = z1
        return [dz1,dz2]

    def h(self,x,u):
        return x