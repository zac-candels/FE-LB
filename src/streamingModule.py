import fenics as fe
from petsc4py import PETSc

class StreamingOperator:

    def __init__(self, V, state, lattice, dt):

        self.Q = lattice.Q
        self.xi = lattice.xi
        self.dt = dt

        self.sysMatStream = []
        self.sysMatLumped = []

        self.advectionMats = []
        self.doubleAdvectionMats = []
        
        self.streamingPrevTimeVecs = [state.f_star_coll[0].vector().copy()\
                                      for _ in range(self.Q)]
        self.advectionVecs = [state.f_star_coll[0].vector().copy()\
                              for _ in range(self.Q)]
        self.doubleAdvectionVecs =[state.f_star_coll[0].vector().copy()\
                                   for _ in range(self.Q)]
        self.rhsVecStreaming = [state.f_star_coll[0].vector().copy()\
                                for _ in range(self.Q)]
            
        self.rhsVecStreamingPetsc = [fe.as_backend_type(
            self.rhsVecStreaming[i]).vec() for i in range(self.Q)]
        self.blockRhsVecStreaming = PETSc.Vec().createNest(self.rhsVecStreamingPetsc)


        self.solverList = []
        
        self.mass_mat = None
        
        self.M_lumped = None

        self._assembleMatrices(V, state)
        



    def _assembleMatrices(self, V, state):

        bilinear_forms = []
        advection_forms = []
        double_advection_forms = []


        for i in range(self.Q):

            bilinear_forms.append(state.f_trial * state.v * fe.dx)

            advection_forms.append(
                state.v
                * fe.dot(self.xi[i], fe.grad(state.f_trial))
                * fe.dx
            )

            double_advection_forms.append(
                fe.dot(self.xi[i], fe.grad(state.v))
                *
                fe.dot(self.xi[i], fe.grad(state.f_trial))
                * fe.dx
            )


        mass_form = (
            state.f_trial
            * state.v
            * fe.dx
        )


        self.mass_mat = fe.assemble(mass_form)


        mass_action = fe.action(
            mass_form,
            fe.Constant(1)
        )


        self.M_lumped = fe.assemble(mass_form)

        self.M_lumped.zero()

        self.M_lumped.set_diagonal(
            fe.assemble(mass_action)
        )


        M_vect = fe.assemble(mass_action)

        M_petsc = fe.as_backend_type(
            M_vect
        ).vec()


        for i in range(self.Q):

            self.sysMatStream.append(
                fe.assemble(
                    bilinear_forms[i]
                )
            )


            self.sysMatLumped.append(
                M_petsc.copy()
            )


            self.advectionMats.append(
                fe.assemble(
                    advection_forms[i]
                )
            )


            self.doubleAdvectionMats.append(
                fe.assemble(
                    double_advection_forms[i]
                )
            )


            self.solverList.append(
                fe.LUSolver(
                    self.sysMatStream[i]
                )
            )
        
    def assembleRhsLumping(self, f_star_coll, dt):
        
        for idx in range(self.Q):
            self.M_lumped.mult(
                f_star_coll[idx].vector(),
                self.streamingPrevTimeVecs[idx])
            
            self.advectionMats[idx].mult(f_star_coll[idx].vector(),
                                    self.advectionVecs[idx])
            
            self.doubleAdvectionMats[idx].mult(f_star_coll[idx].vector(),
                                          self.doubleAdvectionVecs[idx])
    
            self.rhsVecStreaming[idx].zero()
            self.rhsVecStreaming[idx].axpy(1.0,
                                                self.streamingPrevTimeVecs[idx])
            self.rhsVecStreaming[idx].axpy(-dt,
                                                self.advectionVecs[idx])
            self.rhsVecStreaming[idx].axpy(0.5*dt**2,
                                                self.doubleAdvectionVecs[idx])
        return None
    
    def solveSysLumping(self, f_star_stream):
        subvecs = self.blockRhsVecStreaming.getNestSubVecs()
       
        # Solve linear system for streaming step
        for idx in range(self.Q):
            #solver_list[idx].solve(f_nP1[idx].vector(), rhsVecStreaming[idx])
            vi = fe.as_backend_type(self.rhsVecStreaming[idx]).vec()
            f_star_stream[idx].vector().vec().pointwiseDivide(
                vi, 
                self.sysMatLumped[idx])
            
        return f_star_stream