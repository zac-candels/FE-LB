import fenics as fe


def create_bounceback_bcs(V, f_n, L_y, tol=1e-8):


    # Lower wall
    def lower_wall(x, on_boundary):
        return (on_boundary and fe.near(x[1], 0.0, tol) )


    # Upper wall
    def upper_wall(x, on_boundary):
        return (on_boundary and fe.near(x[1], L_y, tol))


    # Lower wall:
    # incoming distributions 5,2,6
    # replaced by outgoing 7,4,8

    lower_pairs = {
        5: 7,
        2: 4,
        6: 8
    }

    upper_pairs = {
        7: 5,
        4: 2,
        8: 6
    }


    bcs = {}

    for incoming, outgoing in lower_pairs.items():

        f_boundary = fe.Function(V)

        fe.project(f_n[outgoing], V, function=f_boundary)

        bcs[incoming] = fe.DirichletBC(V, f_boundary, lower_wall)


    for incoming, outgoing in upper_pairs.items():

        f_boundary = fe.Function(V)

        fe.project(f_n[outgoing], V, function=f_boundary)

        bcs[incoming] = fe.DirichletBC(V, f_boundary, upper_wall)


    return bcs