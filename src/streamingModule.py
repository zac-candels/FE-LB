import sys
sys.path.insert(0, "/home/zcandels/refactor/src")
import fenics as fe
import numpy as np
import moments


class StreamingOperator:

    def __init__(self, V, state, lattice,
                 dt, lumpingOrDirect, forceInCollisionStreaming):

        self.Q = lattice.Q
        self.xi = lattice.xi
        self.w = lattice.weights
        self.dt = dt
        self.c_s = np.sqrt(1/3)
        self.v = state.v
        self.lumpingOrDirect = lumpingOrDirect
        self.forceInCollisionStreaming = forceInCollisionStreaming

        self.sysMatStream = []
        self.sysMatLumped = []

        self.advectionMats = []
        self.doubleAdvectionMats = []
        
        self.streamingPrevTimeVecs = [state.f_star[0].vector().copy()\
                                      for _ in range(self.Q)]
        self.advectionVecs = [state.f_star[0].vector().copy()\
                              for _ in range(self.Q)]
        self.doubleAdvectionVecs =[state.f_star[0].vector().copy()\
                                   for _ in range(self.Q)]
        self.rhsVecStreaming = [state.f_star[0].vector().copy()\
                                for _ in range(self.Q)]


        self.linearForms = []
        
        self.linear_forms_stream = []

        self.solverList = []
        
        self.massMat = None
        
        self.M_lumped = None
        
        self.M_petsc = None

        self._assembleMatrices(V, state)
        
        
    def bodyForce(self, velStar, idx, forceDensity):
        prefactor = self.w[idx]
        inverse_cs2 = 1 / self.c_s**2
        inverse_cs4 = 1 / self.c_s**4

        xi_dot_prod_F = self.xi[idx][0]*forceDensity[0]\
            + self.xi[idx][1]*forceDensity[1]

        u_dot_prod_F = velStar[0]*forceDensity[0] + velStar[1]*forceDensity[1]

        xi_dot_u = self.xi[idx][0]*velStar[0] + self.xi[idx][1]*velStar[1]

        Force = prefactor*(inverse_cs2*(xi_dot_prod_F - u_dot_prod_F)
                           + inverse_cs4*xi_dot_u*xi_dot_prod_F)

        return Force
        
    def createLinearformsForceInStreaming(self, state, dt, forceDensity):
        
        for idx in range(self.Q):
            dot_product_force_term = 0.5*dt**2 * fe.dot(self.xi[idx],
                                                        fe.grad(state.v))\
                * self.bodyForce(state.vel_star, idx, forceDensity) * fe.dx
    
            lin_form_stream = dt*state.v*self.bodyForce(state.vel_star,
                                              idx, forceDensity)*fe.dx\
                + dot_product_force_term
    
            self.linear_forms_stream.append(lin_form_stream)
        
        return None
        
        

    def _assembleMatrices(self, V, state):

        bilinear_forms = []
        advection_forms = []
        double_advection_forms = []

        for idx in range(self.Q):

            bilinear_forms.append(state.f_trial * state.v * fe.dx)

            advection_forms.append(
                state.v
                * fe.dot(self.xi[idx], fe.grad(state.f_trial))
                * fe.dx)

            double_advection_forms.append(
                fe.dot(self.xi[idx], fe.grad(state.v))
                *
                fe.dot(self.xi[idx], fe.grad(state.f_trial))
                * fe.dx)
            

                #double_dot_product_term = -0.5*dt**2 * fe.dot(xi[idx], fe.grad(f_star[idx]))\
                #    * fe.dot(xi[idx], fe.grad(v)) * fe.dx




        mass_form = (
            state.f_trial
            * state.v
            * fe.dx)


        self.massMat = fe.assemble(mass_form)


        mass_action = fe.action(
            mass_form,
            fe.Constant(1))


        self.M_lumped = fe.assemble(mass_form)

        self.M_lumped.zero()

        self.M_lumped.set_diagonal(
            fe.assemble(mass_action))


        M_vect = fe.assemble(mass_action)

        self.M_petsc = fe.as_backend_type(M_vect).vec()


        for i in range(self.Q):

            self.sysMatStream.append(fe.assemble(bilinear_forms[i]))


            self.sysMatLumped.append(self.M_petsc.copy())


            self.advectionMats.append(fe.assemble(advection_forms[i]))


            self.doubleAdvectionMats.append(fe.assemble(
                double_advection_forms[i]))


            self.solverList.append(fe.LUSolver(self.sysMatStream[i]))
            

        
    def assembleRhsLumping(self, f_star, dt, forceDensity):

        for idx in range(self.Q):
            self.M_lumped.mult(
                f_star[idx].vector(),
                self.streamingPrevTimeVecs[idx])
            
            self.advectionMats[idx].mult(f_star[idx].vector(),
                                    self.advectionVecs[idx])
            
            self.doubleAdvectionMats[idx].mult(f_star[idx].vector(),
                                          self.doubleAdvectionVecs[idx])
    
            self.rhsVecStreaming[idx].zero()
            self.rhsVecStreaming[idx].axpy(1.0,
                                                self.streamingPrevTimeVecs[idx])
            self.rhsVecStreaming[idx].axpy(-dt,
                                                self.advectionVecs[idx])
            self.rhsVecStreaming[idx].axpy(0.5*dt**2,
                                                self.doubleAdvectionVecs[idx])
            
            if self.forceInCollisionStreaming == "streaming":
                vel_star = moments.getVel(f_star, self.xi, forceDensity, dt)
                u_dot_prod_F = fe.dot(vel_star, forceDensity)
                xi_dot_prod_F = fe.dot( self.xi[idx], forceDensity)

                xi_dot_u = fe.dot(self.xi[idx], vel_star)
            
                Force = self.w[idx]*( (1/self.c_s**2)*(xi_dot_prod_F - u_dot_prod_F)
                               + (1/self.c_s**4)*xi_dot_u*xi_dot_prod_F)
            
                advectionForceTerm = fe.assemble(
                    fe.dot(self.xi[idx], fe.grad(self.v))* Force * fe.dx)
                    
                basicForceTerm = fe.assemble(self.v*Force*fe.dx)
                
                self.rhsVecStreaming[idx].axpy(dt, basicForceTerm)
                self.rhsVecStreaming[idx].axpy(0.5*dt**2, advectionForceTerm)

            
                
        
        return None
    

    def solveSysLumping(self, f_nP1):

        # Solve linear system for streaming step
        for idx in range(self.Q):
            #solver_list[idx].solve(f_nP1[idx].vector(), rhsVecStreaming[idx])
            vi = fe.as_backend_type(self.rhsVecStreaming[idx]).vec()
            f_nP1[idx].vector().vec().pointwiseDivide(
                vi, 
                self.sysMatLumped[idx])
            
        return f_nP1