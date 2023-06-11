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
    

if __name__ == '__main__':
    f_ode = odeCasADiUnbalancedDisc()
    expr_rk4, x0, u0 = RK4(f_ode, dt=0.1)

    Jfx = Function("Jfx", [x0, u0], [jacobian(expr_rk4,x0)])

    print(expr_rk4.shape)