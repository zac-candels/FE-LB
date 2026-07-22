import fenics as fe

def create_mesh(dim, L_x, L_y, nx, ny):

    comm = fe.MPI.comm_world

    if dim == 2:
        mesh = fe.RectangleMesh(
            comm,
            fe.Point(0.0, 0.0),
            fe.Point(L_x, L_y),
            nx,
            ny
        )
    
        # Periodic boundary in x direction
        class PeriodicBoundaryX(fe.SubDomain):
    
            def inside(self, x, on_boundary):
                return (
                    fe.near(x[0], 0.0)
                    and on_boundary
                )
    
            def map(self, x, y):
                y[0] = x[0] - L_x
                y[1] = x[1]


    pbc = PeriodicBoundaryX()

    V = fe.FunctionSpace(mesh, "P", 1, constrained_domain=pbc)

    Vvec = fe.VectorFunctionSpace(mesh, "P", 1, constrained_domain=pbc)

    return mesh, V, Vvec