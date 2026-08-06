import sys
sys.path.insert(0, "/home/zcandels/refactor/src")
import fenics as fe 
import moments
import numpy as np

class allenCahn:
    
    def __init__(self, V, xc, yc, initDropDiam, interfaceThickness,
                 M_tilde, beta_mass_diff, A, kappa,
                 theta):
        
        self.phi_trial = fe.TrialFunction(V)
        self.mu_trial = fe.TrialFunction(V)
        self.v = fe.TestFunction(V)
        self.M_tilde = M_tilde 
        self.beta_mass_diff = beta_mass_diff 
        self.A = A
        self.kappa = kappa
        self.theta = theta
        self.interfaceThickness = np.sqrt(self.kappa/self.A)
        
        self.mass_diff = fe.Constant(0.0)
        
        c_init_expr = fe.Expression(
            "-tanh( (sqrt(pow(x[0]-xc,2) + pow(x[1]-yc,2)) - R) / (sqrt(2)*eps) )",
            degree=2,  # polynomial degree used for interpolation
            xc=xc,
            yc=yc,
            R=initDropDiam/2,
            eps=interfaceThickness
        )
        
        self.phi_n = fe.interpolate(c_init_expr, V)
        self.phi_nP1 = fe.Function(V)
        self.mu_n = fe.Function(V)
        self.prevTimeAcVec = self.mu_n.vector().copy()
        self.rhsVecTemp = self.mu_n.vector().copy()
        
        bilin_form_AC = self.phi_trial * self.v * fe.dx
        bilin_form_mu = self.mu_trial * self.v * fe.dx
        
        massForm = bilin_form_AC
        mass_action_form = fe.action(massForm, fe.Constant(1))
        M_lumped = fe.assemble(massForm)
        M_lumped.zero()
        M_lumped.set_diagonal(fe.assemble(mass_action_form))
        M_vect = fe.assemble(mass_action_form)
        M_petsc = fe.as_backend_type(M_vect).vec()

    def assembleRhsPhiLumping (self, f_n,
                               xi, dt, forceDensityTuple,
                               M_petsc):
        
        lin_form_AC = (
            -dt * self.v
            * fe.dot(moments.getVel(f_n, xi, forceDensityTuple, dt),
                     fe.grad(self.phi_n)) * fe.dx
        
            - dt * self.M_tilde * self.v * self.mu_n * fe.dx
        
            - (self.beta_mass_diff / dt) * self.mass_diff
            * fe.sqrt(fe.dot(fe.grad(self.phi_n),
                             fe.grad(self.phi_n)))
            * self.v * fe.dx
        
            - 0.5 * dt**2
            * fe.dot(moments.getVel(f_n, xi, forceDensityTuple, dt),
                     fe.grad(self.v))
            * fe.dot(moments.getVel(f_n, xi, forceDensityTuple, dt),
                     fe.grad(self.phi_n))
            * fe.dx)
        
        fe.as_backend_type(self.prevTimeAcVec).vec().pointwiseMult(
            self.phi_n.vector().vec(), M_petsc)
        
        fe.assemble(lin_form_AC, tensor=self.rhsVecTemp)
        rhs_AC = self.prevTimeAcVec + self.rhsVecTemp
        
        return rhs_AC


    def assembleRhsMu(self, ds_bottom):
        lin_form_mu =  ( self.A*self.phi_n
        *(self.phi_n**2 - 1)*self.v*fe.dx\
            
        + self.kappa
        *fe.dot(fe.grad(self.phi_n),fe.grad(self.v))*fe.dx\
            
        + self.kappa/(np.sqrt(2)*self.interfaceThickness)
        *np.cos(self.theta)*(self.phi_n**2-1)*self.v*ds_bottom )
           
        rhs_mu = fe.assemble(lin_form_mu)
        
        return rhs_mu
    
    def solvePhi(self, rhs_AC, M_petsc):
        
        rhsPhiVec = fe.as_backend_type(rhs_AC).vec()
        
        self.phi_nP1.vector().vec().pointwiseDivide(
            rhsPhiVec, M_petsc)
        
        return self.phi_nP1
        
    def solveMu(self, rhs_mu, M_petsc):    
        
        rhsMuVec = fe.as_backend_type(rhs_mu).vec()
        #mu_solver.solve(mu_nP1.vector(), rhs_mu)
        self.mu_n.vector().vec().pointwiseDivide(rhsMuVec, M_petsc)
        
        return self.mu_n
        
        