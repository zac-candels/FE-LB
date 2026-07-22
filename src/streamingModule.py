import fenics as fe


class StreamingOperator:

    def __init__(self, V, state, lattice, dt):

        self.Q = lattice.Q
        self.xi = lattice.xi
        self.dt = dt

        self.sysMatStream = []
        self.sysMatLumped = []

        self.advectionMats = []
        self.doubleAdvectionMats = []

        self.solverList = []
        
        self.mass_mat = None
        
        self.M_lumped = None

        self._assemble(V, state)
        



    def _assemble(self, V, state):

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