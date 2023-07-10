from casadi import *

def RK4(odeCasADi, dt):
    x0 = MX.sym("x0",2,1)
    u0 = MX.sym("u0",1,1)

    k1 = dt*odeCasADi(x0,u0)
    k2 = dt*odeCasADi(x0+1*k1/2,u0)
    k3 = dt*odeCasADi(x0+1*k2/2,u0)
    k4 = dt*odeCasADi(x0+1*k3,u0)

    expr_rk4 = x0 + (k1+2*k2+2*k3+k4)/6
    # f_rk4 = Function('f_rk4', [x0, u0], [expr_rk4])

    return expr_rk4, x0, u0


def odeCasADiUnbalancedDisc():
    M = 0.0761844495320390 # mass of the cart [kg] -> now estimated
    g = 9.80155078791343 # gravity constant [m/s^2]
    J = 0.000244210523960356
    Km = 10.5081817407479
    I = 0.0410772235841364
    tau = 0.397973147009910

    # set up states & controls
    theta   = SX.sym('theta')
    dtheta  = SX.sym('dtheta')

    x = vertcat(dtheta, theta)
    u = SX.sym('u')

    # dynamics
    expr_ode = vertcat(-M*g*I/J*sin(theta) - 1/tau*dtheta + Km/tau*u, dtheta)
    f_ode = Function('f_ode', [x, u], [expr_ode])

    return f_ode

def odeCasADiMassSpringDamper():
    k=1e4; c=5; m=0.25

    # set up states & controls
    dx   = SX.sym('theta')
    x  = SX.sym('dtheta')

    x_cas = vertcat(dx, x)
    u_cas = SX.sym('u')

    # dynamics
    expr_ode = vertcat(-k/m*x - c/m*dx + 1/m*u_cas, dx)
    f_ode = Function('f_ode', [x_cas, u_cas], [expr_ode])

    return f_ode
    
class CasADi_model():
    def __init__(self, ode, h_ix, dt, nx, nu):
        self.dt = dt
        self.ode = ode
        self.ny = len(h_ix) if type(h_ix) is not int else 1
        self.nx = nx
        self.nu = nu

        self.expr_rk4, self.x_cas, self.u_cas = self.RK4()

        if type(h_ix) == int:
            self.expr_output = vertcat(self.x_cas[h_ix])
        else:
            self.expr_output = vertcat(self.x_cas[h_ix[0]])
            if len(h_ix) != 1:
                for i in range(1,len(h_ix)):
                    self.expr_output = vertcat(self.expr_output, self.x_cas[h_ix[i]])

    def RK4(self):
        x0 = MX.sym("x0",self.nx,1)
        u0 = MX.sym("u0",self.nu,1)

        k1 = self.dt*self.ode(x0,u0)
        k2 = self.dt*self.ode(x0+1*k1/2,u0)
        k3 = self.dt*self.ode(x0+1*k2/2,u0)
        k4 = self.dt*self.ode(x0+1*k3,u0)

        expr_rk4 = x0 + (k1+2*k2+2*k3+k4)/6

        return expr_rk4, x0, u0

if __name__ == '__main__':
    f_ode = odeCasADiMassSpringDamper()
    expr_rk4, x0, u0 = RK4(f_ode, dt=0.002)
    model = CasADi_model(f_ode, (0,1), dt=0.002, nx=2, nu=1)

    Jfx = Function("Jfx", [x0, u0], [jacobian(model.expr_rk4,x0)])
    Jhx = Function("Jhx", [x0, u0], [jacobian(model.expr_output,x0)])

    print(u0.size()[0])