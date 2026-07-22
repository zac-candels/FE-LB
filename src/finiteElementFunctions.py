import fenics as fe


class SimulationState:

    def __init__(self, V, Vvec, Q):

        # velocity fields
        self.vel_n = fe.Function(Vvec)
        self.vel_star = fe.Function(Vvec)


        # distributions
        self.f_n = []

        for i in range(Q):
            self.f_n.append(fe.Function(V))

        self.f_nP1 = []

        for i in range(Q):
            self.f_nP1.append(fe.Function(V))

        self.f_star_coll = []

        for i in range(Q):
            self.f_star_coll.append(fe.Function(V))

        self.f_star_stream = []

        for i in range(Q):
            self.f_star_stream.append(fe.Function(V))

        # density
        self.rho_n = fe.Function(V)

        # variational objects
        self.f_trial = fe.TrialFunction(V)
        self.v = fe.TestFunction(V)