import fenics as fe
import os
import numpy as np
#import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from mpi4py import MPI
comm = fe.MPI.comm_world
rank = fe.MPI.rank(comm)

plt.close('all')

# Where to save the plots
WORKDIR = os.getcwd()

T = 4
L_x, L_y = 1, 1
nx, ny = 128, 128
h = L_x/nx
dt = 0.1*h
num_steps = int(np.ceil(T/dt))

M_tilde = 0.005

outDirName = f"M_{M_tilde}_parallel"

if rank == 0:
    os.makedirs(outDirName, exist_ok=True)
comm.Barrier()


eps = 0.7*h

center_init_x, center_init_y = L_x/2, L_y/2


mesh = fe.RectangleMesh(comm, fe.Point(0, 0), fe.Point(L_x, L_y), nx, ny)

class PeriodicBoundary2D(fe.SubDomain):
    # Define the "master boundary": x=0 or y=0
    def inside(self, x, on_boundary):
        # This is true for points on the left or bottom edges (master edges)
        return ( (fe.near(x[0], 0) or fe.near(x[1], 0)) and
                 on_boundary and
                 not ((fe.near(x[0], 0) and fe.near(x[1], L_y)) or
                      (fe.near(x[0], L_x) and fe.near(x[1], 0))) )

    # Map slave boundary points onto the master boundary
    def map(self, x, y):
        if fe.near(x[0], L_x) and fe.near(x[1], L_y):
            # Top-right corner → bottom-left corner
            y[0] = x[0] - L_x
            y[1] = x[1] - L_y
        elif fe.near(x[0], L_x):
            # Right boundary → left boundary
            y[0] = x[0] - L_x
            y[1] = x[1]
        elif fe.near(x[1], L_y):
            # Top boundary → bottom boundary
            y[0] = x[0]
            y[1] = x[1] - L_y

pbc = PeriodicBoundary2D()

V = fe.FunctionSpace(mesh, "P", 1, constrained_domain=pbc)

# Define trial and test functions, as well as
# finite element functions at previous timesteps

f_trial = fe.TrialFunction(V)

phi_n = fe.Function(V)
V_vec = fe.VectorFunctionSpace(mesh, "P", 1, constrained_domain=pbc)
vel_n = fe.Function(V_vec)
mu_n = fe.Function(V)
mu_nP1 = fe.Function(V)

v = fe.TestFunction(V)

phi_nP1 = fe.Function(V)

u_expr = fe.Expression(
    (
        "2*sin(pi*x[0])*sin(pi*x[0])*sin(pi*x[1])*cos(pi*x[1])*cos(pi*t/T)",
        "-2*sin(pi*x[0])*cos(pi*x[0])*sin(pi*x[1])*sin(pi*x[1])*cos(pi*t/T)"
    ),
    degree=5, pi=np.pi, T=T, t=0.0
)

# Project onto vector finite element space (periodic in y via same constrained_domain)
vel_n = fe.project(u_expr, V_vec)

# Define Allen-Cahn mobility
def mobility(phi_n):
    grad_phi_n = fe.grad(phi_n)

    abs_grad_phi_n = fe.sqrt(fe.dot(grad_phi_n, grad_phi_n) + 1e-6)
    inv_abs_grad_phi_n = 1.0 / abs_grad_phi_n

    mob = M_tilde( 1 - 4*phi_n*(1 - phi_n)/eps * inv_abs_grad_phi_n )
    return mob


# Initialize \phi
phi_init_expr = fe.Expression(
    "0.5 - 0.5 * tanh( (sqrt(pow(x[0]-xc,2) + pow(x[1]-yc,2)) - R) / (2*eps) ) ",
    degree=4,  # polynomial degree used for interpolation
    xc=0.5,
    yc=0.75,
    R=0.15,
    eps=eps
)

phi_n = fe.project(phi_init_expr, V)


bilin_form_AC = f_trial * v * fe.dx

lin_form_AC = phi_n * v * fe.dx - dt*v*fe.dot(vel_n, fe.grad(phi_n))*fe.dx\
    - dt*fe.dot(fe.grad(v), mobility(phi_n)*fe.grad(phi_n))*fe.dx\
        - 0.5*dt**2 * fe.dot(vel_n, fe.grad(v)) * fe.dot(vel_n, fe.grad(phi_n)) *fe.dx\
           


phi_mat = fe.assemble(bilin_form_AC)

phi_solver = fe.KrylovSolver("cg", "hypre_amg")
phi_solver.set_operator(phi_mat)
prm = phi_solver.parameters
prm["absolute_tolerance"] = 1e-14
prm["relative_tolerance"] = 1e-8
prm["maximum_iterations"] = 1000
prm["nonzero_initial_guess"] = False

outfile = fe.XDMFFile(comm, f"{outDirName}/solution.xdmf")
outfile.parameters["flush_output"] = True
outfile.parameters["functions_share_mesh"] = True
outfile.parameters["rewrite_function_mesh"] = False

# Timestepping
t = 0.0
for n in range(num_steps):
    t += dt
    
    u_expr.t = t
    vel_n.assign(fe.project(u_expr, V_vec))

        
    # rho_post = np.sum(f_post_stack, axis=0)
    # momx_post = np.sum(f_post_stack * xi_array[:, 0][:, None], axis=0)
    # momy_post = np.sum(f_post_stack * xi_array[:, 1][:, None], axis=0)
    
        
    rhs_AC = fe.assemble(lin_form_AC)
    
    phi_solver.solve(phi_nP1.vector(), rhs_AC)


    # Update previous solutions

    phi_n.assign(phi_nP1)


    
    if n % 10 == 0:  # plot every 10 steps
    
        #c_n.rename("phi", "phase field")  # Optional: label for visualization
        #xdmf_file.write(c_n, t)
        
        outfile.write(phi_n, t)
        





