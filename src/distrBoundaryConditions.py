import fenics as fe


class BounceBackBoundary:

    def __init__(self, V, f_n, L_y, tol=1e-8):

        self.V = V
        self.f_n = f_n

        self.lower = {}
        self.upper = {}

        # Functions that hold the boundary values
        self.f5_lower_func = fe.Function(V)
        self.f2_lower_func = fe.Function(V)
        self.f6_lower_func = fe.Function(V)

        self.f7_upper_func = fe.Function(V)
        self.f4_upper_func = fe.Function(V)
        self.f8_upper_func = fe.Function(V)

        self._create_bcs(L_y, tol)


    def _create_bcs(self, L_y, tol):

        def lower_wall(x, on_boundary):
            return (on_boundary and fe.near(x[1], 0.0, tol))

        def upper_wall(x, on_boundary):
            return (on_boundary and fe.near(x[1], L_y, tol))


        self.lower["f5"] = fe.DirichletBC(self.V, self.f5_lower_func, lower_wall)

        self.lower["f2"] = fe.DirichletBC(
            self.V,
            self.f2_lower_func,
            lower_wall
        )

        self.lower["f6"] = fe.DirichletBC(
            self.V,
            self.f6_lower_func,
            lower_wall
        )


        self.upper["f7"] = fe.DirichletBC(
            self.V,
            self.f7_upper_func,
            upper_wall
        )

        self.upper["f4"] = fe.DirichletBC(
            self.V,
            self.f4_upper_func,
            upper_wall
        )

        self.upper["f8"] = fe.DirichletBC(
            self.V,
            self.f8_upper_func,
            upper_wall
        )


    def update(self, f_star_coll):

        self.f5_lower_func.assign(
            f_star_coll[7]
        )

        self.f2_lower_func.assign(
            f_star_coll[4]
        )

        self.f6_lower_func.assign(
            f_star_coll[8]
        )


        self.f7_upper_func.assign(
            f_star_coll[5]
        )

        self.f4_upper_func.assign(
             f_star_coll[2]
        )

        self.f8_upper_func.assign(
             f_star_coll[6]
        )