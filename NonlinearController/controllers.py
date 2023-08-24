import numpy as np
from NonlinearController.model_utils import *
from NonlinearController.mpc_utils import *
from NonlinearController.utils import *
from NonlinearController.lpv_embedding import *
from NonlinearController.models import *
import qpsolvers as qp

class VelocityMpcController():
    """
    A class for velocity MPC controller.

    ...

    Attributes
    ----------
    ph : str
        placeholder

    Methods
    -------
    __call__(reference):
        Returns optimal input for given reference and progresses controller by one step.
    """

    def __init__(self, system, model, Nc, Q1, Q2, R, P, qlim, wlim, nr_sim_steps, max_iter=1, n_stages=1, numerical_method=1, model_simulation="LPV", dev=None, factorization_method=None):
        """
        Constructs all the necessary attributes for the person object.

        Parameters
        ----------
            system : deepSI.System_deriv
                continuous system with rk4 discretization for timesteps
            model : CasADi_model (or) deepSI.fit_systems.SS_encoder_general
                discrete model approximating dynamics of system
            Nc : int
                control horizon of mpc controller
            Q1 : np.ndarray
                weighting matrix for output
            Q2 : np.ndarray
                weighting matrix for incremental states
            R : np.ndarray
                weighting matrix for incremental inputs
            P : np.ndarray
                weighting matrix for difference real setpoint and artificial setpoint
            ylim : np.ndarray

        """

        self.system = system
        self.dt = system.dt

        self.nu = system.nu if system.nu is not None else 1
        self.ny = system.ny if system.ny is not None else 1
        self.ne = 1

        self.dev = dev

        self.model = model
        if type(self.model) == CasADi_model:
            self.model_type = "CasADi"
        else:
            self.model_type = "encoder"
        self.nx = model.nx
        self.nz = self.nx+self.ny

        if factorization_method == None:
            self.factorization_method = self.model_type
        else:
            self.factorization_method = factorization_method
        print("Factorization method: " + self.factorization_method)

        self.model_simulation = model_simulation

        self.Nc = Nc

        self.Q1 = Q1
        self.Q2 = Q2
        self.R = R
        self.P = P

        self.max_iter = max_iter
        self.numerical_method = numerical_method
        self.n_stages = n_stages

        self.w_max = wlim; self.w_min = -wlim
        self.q_max = [qlim]; self.q_min = [-qlim]
        self.w0 = 0; self.q0 = [0.0]

        self.Offline_QP()

        # logging
        self.nr_sim_steps = nr_sim_steps
        self.sim_step = 0
        self.log_solv_t = np.zeros(self.nr_sim_steps)
        self.log_fact_t = np.zeros(self.nr_sim_steps)
        self.log_comp_t = np.zeros(self.nr_sim_steps)
        self.log_iterations = np.zeros(self.nr_sim_steps)

    def Offline_QP(self):
        """
        Initiates the variables and objects required for the QP solving online.
        """

        self.Omega1 = get_Omega(self.Nc, self.Q1)
        self.Omega2 = get_Omega(self.Nc, self.Q2)
        self.Psi = get_Psi(self.Nc, self.R)

        # extended objective matrices for soft constraints
        e_lambda = 1e10 # weighting of minimizing e in objective function
        self.Ge = np.zeros((self.Nc*self.nu+self.ne,self.Nc*self.nu+self.ne)) 
        self.Ge[-self.ne:,-self.ne:] = e_lambda

        if self.model_type == "encoder":
            if self.factorization_method == "encoder":
                self.embedder = velocity_lpv_embedder_autograd(ss_enc=self.model, Nc=self.Nc, n_stages=self.n_stages, numerical_method=self.numerical_method, dev=self.dev)
            elif self.factorization_method == "CasADi":
                self.embedder = velocity_lpv_embedder(ss_enc=self.model, Nc=self.Nc, n_stages=self.n_stages, numerical_method=self.numerical_method)
        else:
            self.embedder = CasADi_velocity_lpv_embedder(model=self.model, Nc=self.Nc, n_stages=self.n_stages, numerical_method=self.numerical_method)

        # normalize initial input and output
        if self.model_type == "encoder":
            self.norm = self.model.norm
        else:
            self.norm = normalizer(np.array([0.0]), np.array([1.0]), 0, 1)
        self.u0 = norm_input(self.w0, self.norm)
        self.y0 = norm_output(self.q0, self.norm)

        # determine constraint matrices
        self.u_min = norm_input(self.w_min, self.norm)
        self.u_max = norm_input(self.w_max, self.norm)
        self.y_min = np.hstack((norm_output(self.q_min, self.norm), np.ones(self.nx)*-1000))
        self.y_max = np.hstack((norm_output(self.q_max, self.norm), np.ones(self.nx)*1000)) # augmented with the velocity states
        self.D, self.E, self.M, self.c = getDEMc(self.y_min, self.y_max, self.u_min, self.u_max, self.Nc, self.nz, self.nu)
        self.Lambda = np.tril(np.ones((self.Nc,self.Nc)),0)

        # determine terminal constraint matrices
        self.M_terminal = np.zeros((self.nx+self.ny,self.ny)); self.M_terminal[:self.ny,:] = np.eye(self.ny)
        self.E_terminal = np.zeros((self.nx+self.ny,self.nx)); self.E_terminal[self.ny:,:] = np.eye(self.nx)
        self.Sy = np.hstack((np.zeros((self.ny,(self.Nc-1)*self.ny)),np.eye(self.ny)))
        # self.Sx = np.hstack((np.zeros((self.nx,(self.Nc-1)*self.nz+self.ny)),np.eye(self.nx)))
        Sx = np.zeros((self.nz,self.nz)); Sx[self.ny:, self.ny:] = np.eye(self.nx); self.Sx = np.kron(np.eye(self.Nc),Sx)

        # initial predicted states, input, and output
        if self.model_type == "encoder":
            self.nb = self.model.nb
            self.uhist = torch.ones((1,self.nb))*self.u0
            self.na = self.model.na
            self.yhist = torch.Tensor(self.y0[np.newaxis].T).repeat(1,self.na+1)[None,:,:]
            self.X_1 = np.tile(self.model.encoder(self.uhist,self.yhist).detach().numpy(),self.Nc+2).T
        else:
            self.X_1 = np.tile(np.zeros((1,2)),self.Nc+2).T # This cannot be assumed always. Depends on initial conditions of system
        self.U_1 = np.ones((self.Nc+1)*self.nu)[np.newaxis].T*self.u0
        self.Y_1 = np.tile(self.y0[np.newaxis],self.Nc).T

        # run embedding with initial points to load model for embedding
        if self.model_type == "encoder":
            for i in range(5):
                self.embedder(self.X_1, self.U_1)

    def __call__(self, reference):
        """
        Placeholder

        Parameters
        ----------
            system : deepSI.System_deriv
                continuous system with rk4 discretization for timesteps

        """

        raise Exception("Not implemented")
    
    def QP_solve(self, reference):
        """
        Returns optimal input for given reference and progresses controller by one step.

        Parameters
        ----------
            reference : np.ndarray
                reference to be tracked by QP of MPC

        """

        reference = self.convertReference(reference)
        reference = norm_output(reference, self.norm)
        r = extendReference(reference, 0, self.ny, self.Nc)
    
        #++++++++++++++++++ start iteration +++++++++++++++++++++++
        for iteration in range(self.max_iter):
            comp_time_start = time.time()

            # determine predicted velocity states and output
            dX0 = differenceVector(self.X_1[:-self.nx], self.nx)
            dU0 = differenceVector(self.U_1, self.nu)
            # determine extended state from predicted output and velocity states
            Z0 = extendState(self.Y_1, dX0, self.nx, self.ny, self.Nc)

            # determine lpv state space dependencies
            fact_time_start = time.time()

            list_A, list_B, list_C = self.embedder(self.X_1, self.U_1)
            list_ext_A, list_ext_B, list_ext_C = extendABC(list_A, list_B, list_C, self.nx, self.ny, self.nu, self.Nc)
            
            self.log_fact_t[self.sim_step] = self.log_fact_t[self.sim_step] + time.time() - fact_time_start

            # describe optimization problem
            Phi = get_Phi(list_ext_A, self.Nc, self.nz)
            Gamma = get_Gamma(list_ext_A, list_ext_B, self.Nc, self.nz, self.nu)
            Z = getZ(list_ext_C,self.Nc,self.ny,self.nz)

            G = 2*(Gamma.T @ (Z.T @ self.Omega1 @ Z + self.Sx.T @ self.Omega2 @ self.Sx) @ Gamma + self.Psi)
            F = 2*(Gamma.T @ (Z.T @ self.Omega1 @ (Z @ Phi @ Z0[:self.nz] - r) + self.Sx.T @ self.Omega2 @ self.Sx @ Phi @ Z0[:self.nz]))

            # G = 2*(Gamma.T @ (Z.T @ (self.Omega1 + self.Sy.T @ self.P @ self.Sy) @ Z + self.Omega2) @ Gamma + self.Psi)
            # F = 2*(Gamma.T @ (Z.T @ (self.Omega1 @ (Z @ Phi @ Z0[:self.nz] - r) + \
            #                          self.Sy.T @ self.P @ (self.Sy @ Z @ Phi @ Z0[:self.nz] - r[-self.ny:])) + self.Omega2 @ Phi @ Z0[:self.nz]))

            # describe inequality constraints
            L = (self.M @ Gamma + self.E @ self.Lambda)
            alpha = np.ones((self.Nc,1))*self.U_1[0,0]
            W = -(self.E @ alpha + (self.D + self.M @ Phi) @ Z0[:self.nz])
            # add soft constraints
            self.Ge[:self.Nc*self.nu, :self.Nc*self.nu] = G
            Fe = np.vstack((F, np.zeros((self.ne,1))))
            Le = np.hstack((L, -np.ones((self.Nc*2*(self.nz+self.nu)+2*self.nz,self.ne))))

            # describe equality constraints
            # self.c_terminal = np.vstack((r[-self.ny:,:],np.zeros((self.nx,1))))
            # A = (self.E_terminal @ self.Sx) @ Gamma
            # b = self.c_terminal-(self.E_terminal @ self.Sx) @ Phi @ Z0[:self.nz]
            # add soft constraints
            # Ae = np.hstack((A,np.zeros((self.nz,1))))

            # solve QP problem
            solve_time_start = time.time()

            # opt_result = qp.solve_qp(G,F,solver="osqp",initvals=np.hstack((dU0[:,0])))
            # opt_result = qp.solve_qp(self.Ge,Fe,solver="osqp",initvals=np.hstack((dU0[:,0],0)))
            opt_result = qp.solve_qp(self.Ge,Fe,Le,self.c+W,solver="osqp",initvals=np.hstack((dU0[:,0],0)))
            # opt_result = qp.solve_qp(P=self.Ge,q=Fe,G=Le,h=self.c+W,solver="osqp",initvals=np.hstack((dU0[:,0],0)))
            # opt_result = qp.solve_qp(P=self.Ge,q=Fe,A=Ae,b=b,solver="osqp",initvals=np.hstack((dU0[:,0],0)))
            # opt_result = qp.solve_qp(P=self.Ge,q=Fe,G=Le,h=(self.c+W)[:,0],A=Ae,b=b[:,0],solver="osqp",initvals=np.hstack((dU0[:,0],0)))

            self.log_solv_t[self.sim_step] = self.log_solv_t[self.sim_step] + time.time() - solve_time_start

            # extract input from solution of QP
            dU0[:,0] = opt_result[:self.Nc*self.nu]

            # save previous iteration of U_1
            U_1_old = np.copy(self.U_1)
            # compute U_1 from dU0 and previous data
            for i in range(1,self.Nc+1):
                self.U_1[(i*self.nu):(i*self.nu+self.nu),:] = dU0[((i-1)*self.nu):((i-1)*self.nu+self.nu),:].copy() \
                    + self.U_1[((i-1)*self.nu):((i-1)*self.nu+self.nu),:].copy()

            if self.model_simulation == "True":
                # simuate X1 with RK4 from U0 computed above
                x_sim = self.X_1[self.nx:self.nx*2,0].copy()
                U_sim = self.U_1[self.nu:,0].copy()
                X1 = np.zeros((self.nx*self.Nc,1))

                for j in range(self.Nc):
                    if self.model_type == "CasADi":
                        x_sim = self.system.f(x_sim, U_sim[j])
                        X1[(j)*self.nx:(j+1)*self.nx,0] = x_sim.copy()
                    elif self.model_type == "encoder":
                        xt = torch.Tensor(x_sim[np.newaxis])
                        ut = torch.Tensor(U_sim[j][np.newaxis][np.newaxis])
                        x_sim = self.model.fn(xt, ut).detach().numpy()[0,:]
                        X1[(j)*self.nx:(j+1)*self.nx,0] = x_sim.copy()
                    else:
                        raise Exception("invalid model type for forward simulation")

                # Determine Y0 and dX1 from X1 and previous data
                dX1 = X1 - np.hstack((self.X_1[self.nx:self.nx*2,0],X1[:-self.nx,0]))[np.newaxis].T
                Y1 = np.hstack(np.split(X1,self.Nc))[1,:][np.newaxis].T
                Y0 = np.vstack((self.Y_1[self.ny:self.ny*2,:], Y1[:-self.ny,:]))
            elif self.model_simulation == "LPV":
                # predict states
                Z1 = Phi @ Z0[:self.nz] + Gamma @ dU0
                # split extended state up into ouputs and velocity states
                Y0, dX1 = decodeState(Z1, self.nx, self.ny, self.Nc)
            else:
                raise Exception("invalid forward simulation method")

            # overwrite previous predicted states and output with new predicted states and output
            self.Y_1[2*self.ny:,0] = Y0[self.ny:-self.ny,0].copy(); dX0[self.nx:,0] = dX1[:-self.nx,0].copy() #change the shifting on the output to be consequential
            
            # determine new X_1 states from known x0 and predicted dX0
            for i in range(2,self.Nc+1):
                self.X_1[(i*self.nx):(i*self.nx+self.nx),:] = dX0[((i-1)*self.nx):((i-1)*self.nx+self.nx),:] \
                    + self.X_1[((i-1)*self.nx):((i-1)*self.nx+self.nx),:]
            self.X_1[-self.nx:,:] = dX1[-self.nx:,:] + self.X_1[-2*self.nx:-self.nx,:]

            self.log_comp_t[self.sim_step] = self.log_comp_t[self.sim_step] + time.time() - comp_time_start

            # stopping condition
            if np.linalg.norm(self.U_1 - U_1_old) < 1e-5:
                break

        self.log_iterations[self.sim_step] = iteration
        self.sim_step += 1

        u0 = self.U_1[self.nu:self.nu*2,0].copy()
        return denorm_input(u0, self.norm)
        
    def convertReference(self, reference):
        """
        Returns reference of correct shape for controller

        Parameters
        ----------
            reference : int (or) np.ndarray
                reference to be tracked by QP of MPC

        """

        # convert references of single value to array for ny==1
        if type(reference) != numpy.ndarray and self.ny==1:
            reference = np.tile(reference,self.Nc)

        # Add extra dimension if possible and required to reference
        if len(reference.shape) == 1 and self.ny == 1:
            reference = reference[np.newaxis]

        # Return error if reference cannot be handled by controller
        if len(reference.shape) > 2:
            raise ValueError("Reference has to many dimensions for controller.")
        
        return reference
    
    def update(self, q1, w0, x1=None):
        """
        Updates class' prediction lists by one timestep

        Parameters
        ----------
            y1 : np.ndarray
                measurement of system
            u0 : np.ndarray
                optimal input applied to system for measurement
            x1 : np.ndarray
                full state measurement of system

        """
        y1 = norm_output(q1, self.norm)
        u0 = norm_input(w0, self.norm)

        if self.model_type == "encoder":
            # shift history input and output for encoder
            for j in range(self.nb-1):
                self.uhist[0,j] = self.uhist[0,j+1]
            self.uhist[0,self.nb-1] = torch.Tensor(u0)
            for j in range(self.na):
                self.yhist[0,:,j] = self.yhist[0,:,j+1]
            self.yhist[0,:,self.na] = torch.Tensor([y1])
            # predict state with encoder
            x1 = self.model.encoder(self.uhist,self.yhist).detach().numpy().T

        self.X_1[:-self.nx, :] = self.X_1[self.nx:, :]; self.X_1[self.nx:2*self.nx, :] = x1.copy()
        self.U_1[:-self.nu, :] = self.U_1[self.nu:, :]
        self.Y_1[:-self.ny, :] = self.Y_1[self.ny:, :]; self.Y_1[self.ny:2*self.ny, :] = y1[np.newaxis].T.copy()

        return
    
    def computationTimeLogging(self):
        return np.max(self.log_comp_t)*1000, np.median(self.log_comp_t)*1000, \
            np.std(self.log_comp_t)*1000, np.median(self.log_solv_t)*1000, np.median(self.log_fact_t)*1000
    
def controlLoop(reference_theta, system, model, Nc, nr_sim_steps, nu, ny, Q1, Q2, R, P, qlim, wlim, max_iter, n_stages, numerical_method, model_simulation):
    system.reset_state()
    log_q = np.zeros((ny,nr_sim_steps))
    log_w = np.zeros((nu,nr_sim_steps))

    controller = VelocityMpcController(system, model, Nc, Q1, Q2, R, P, qlim, wlim, nr_sim_steps=nr_sim_steps, \
                                        max_iter=max_iter, n_stages=n_stages, numerical_method=numerical_method, model_simulation=model_simulation)

    sim_start_time = time.time()

    for k in range(nr_sim_steps):
        w0 = controller.QP_solve(reference_theta[k:k+Nc])
        system.x = system.f(system.x, w0[0])
        omega1, theta1 = system.h(system.x, w0[0])
        q1 = theta1; x1 = np.vstack((omega1, theta1))
        controller.update(q1, w0, x1)

        log_q[:,k] = q1
        log_w[:,k] = w0

    sim_end_time = time.time()
    print("Sim duration: " + str(sim_end_time - sim_start_time))
    print("Time breakdown: " + str(controller.computationTimeLogging()))

    return log_w, log_q

def controlLoopTime(reference_theta, system, model, Nc, nr_sim_steps, nu, ny, Q1, Q2, R, P, qlim, wlim, max_iter, n_stages, numerical_method, model_simulation):
    system.reset_state()
    log_q = np.zeros((ny,nr_sim_steps))
    log_w = np.zeros((nu,nr_sim_steps))

    controller = VelocityMpcController(system, model, Nc, Q1, Q2, R, P, qlim, wlim, nr_sim_steps=nr_sim_steps, \
                                        max_iter=max_iter, n_stages=n_stages, numerical_method=numerical_method, model_simulation=model_simulation)

    sim_start_time = time.time()

    for k in range(nr_sim_steps):
        w0 = controller.QP_solve(reference_theta[k:k+Nc])
        system.x = system.f(system.x, w0[0])
        omega1, theta1 = system.h(system.x, w0[0])
        q1 = theta1; x1 = np.vstack((omega1, theta1))
        controller.update(q1, w0, x1)

        log_q[:,k] = q1
        log_w[:,k] = w0

    sim_end_time = time.time()
    print("Sim duration: " + str(sim_end_time - sim_start_time))
    print("Time breakdown: " + str(controller.computationTimeLogging()))

    return log_w, log_q, controller.log_comp_t