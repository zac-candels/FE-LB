import fenics as fe
import os
import numpy as np
#import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
#from fenics import XDMFFile, File

plt.close('all')
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

fe.parameters["std_out_all_processes"] = False
fe.set_log_level(fe.LogLevel.ERROR)

# Where to save the plots
WORKDIR = os.getcwd()
outDirName = os.path.join(WORKDIR, "singleVortexTG")
os.makedirs(outDirName, exist_ok=True)

T = 4
CFL = 0.2
L_x, L_y = 1, 1
nx, ny = 128, 128
h = L_x/nx
dt = 0.1*h
num_steps = int(np.ceil(T/dt))

gammaBar = 1
eps = 0.7*h

xc=0.5
yc=0.75
R=0.15


# Set up domain. For simplicity, do unit square mesh.
mesh = fe.RectangleMesh(fe.Point(0, 0), fe.Point(L_x, L_y), nx, ny)

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

c_n = fe.Function(V)
V_vec = fe.VectorFunctionSpace(mesh, "P", 1, constrained_domain=pbc)
vel_n = fe.Function(V_vec)
v = fe.TestFunction(V)

c_nP1 = fe.Function(V)

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
def mobility(c_n):
    grad_c_n = fe.grad(c_n)

    abs_grad_c_n = fe.sqrt(fe.dot(grad_c_n, grad_c_n) + 1e-6)
    inv_abs_grad_c_n = 1.0 / abs_grad_c_n

    mob = gammaBar*eps*(1 - c_n*(1-c_n)/eps * inv_abs_grad_c_n)
    return mob


# Initialize \c

tol = eps

c_init_expr = fe.Expression(
    "0.5 + 0.5 * tanh( (sqrt(pow(x[0]-xc,2) + pow(x[1]-yc,2)) - R) / (2*eps) ) ",
    degree=2,  # polynomial degree used for interpolation
    xc=0.5,
    yc=0.75,
    R=0.15,
    eps=eps
)
c_n = fe.project(c_init_expr, V)

xdmf_file = fe.XDMFFile(comm, os.path.join(outDirName, "phi.xdmf"))
xdmf_file.parameters["flush_output"] = True
xdmf_file.parameters["functions_share_mesh"] = True
xdmf_file.parameters["rewrite_function_mesh"] = False

coords = mesh.coordinates()
c_vals = c_n.compute_vertex_values(mesh)
triangles = mesh.cells()  # get mesh connectivity
triang = tri.Triangulation(coords[:, 0], coords[:, 1], triangles)

plt.figure(figsize=(6,5))
plt.tricontourf(triang, c_vals, levels=50, cmap="RdBu_r")
plt.colorbar(label=r"$\phi$")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()

bilin_form_AC = f_trial * v * fe.dx

lin_form_AC = c_n * v * fe.dx - dt*v*fe.dot(vel_n, fe.grad(c_n))*fe.dx\
    - dt*fe.dot(fe.grad(v), mobility(c_n)*fe.grad(c_n))*fe.dx\
        - 0.5*dt**2 * fe.dot(vel_n, fe.grad(v)) * fe.dot(vel_n, fe.grad(c_n)) *fe.dx\



c_mat = fe.assemble(bilin_form_AC)
c_solver = fe.KrylovSolver("cg", "ilu")
c_solver.set_operator(c_mat)
prm = c_solver.parameters
prm["absolute_tolerance"] = 1e-12
prm["relative_tolerance"] = 1e-8
prm["maximum_iterations"] = 1000
prm["nonzero_initial_guess"] = False

vtkfile = fe.File('TG_output/solution.pvd')


t = 0
for n in range(num_steps):
    
    t += dt
    u_expr.t = t

    vel_n.assign(fe.project(u_expr, V_vec))
    
    rhs_AC = fe.assemble(lin_form_AC)
    
    c_solver.solve(c_nP1.vector(), rhs_AC)

    # Update previous solutions
    c_n.assign(c_nP1)
    
    if n % 10 == 0:  # plot every 10 steps
    
        #c_n.rename("phi", "phase field")  # Optional: label for visualization
        #xdmf_file.write(c_n, t)
        
        vtkfile << c_n
        # Or for PVD: pvd_file << (c_n, t)