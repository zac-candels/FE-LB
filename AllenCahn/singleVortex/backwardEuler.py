import fenics as fe
import os
import numpy as np
#import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri

plt.close('all')

#fe.parameters["std_out_all_processes"] = False
#fe.set_log_level(fe.LogLevel.ERROR)

# Where to save the plots
WORKDIR = os.getcwd()
outDirName = os.path.join(WORKDIR, "singleVortexBE")
os.makedirs(outDirName, exist_ok=True)

T = 4
CFL = 0.2
L_x, L_y = 1, 1
nx, ny = 128, 128
h = L_x/nx
dt = h*0.01
num_steps = int(np.ceil(T/dt))


eps = 0.7*h

center_init_x, center_init_y = L_x/2, L_y/2


# Set up domain. For simplicity, do unit square mesh.
mesh = fe.RectangleMesh(fe.Point(0, 0), fe.Point(L_x, L_y), nx, ny)

# Set periodic boundary conditions at left and right endpoints
class PeriodicBoundaryY(fe.SubDomain):
    def inside(self, point, on_boundary):
        # bottom boundary (y=0)
        return fe.near(point[1], 0.0) and on_boundary

    def map(self, x, y):
        # map top boundary (y=Ly) to bottom (y=0)
        # In FEniCS the map signature is map(self, x, y) where x is a point on master boundary,
        # and y is the mapped point; set y[...] accordingly.
        y[0] = x[0]
        y[1] = x[1] - L_y

pbc = PeriodicBoundaryY()

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

    mob = (1/np.pi)*( 1 - 4*c_n*(1 - c_n)/eps * inv_abs_grad_c_n )
    return mob


# Initialize \c
c_init_expr = fe.Expression(
    "0.5 - 0.5 * tanh( 2.0 * (sqrt(pow(x[0]-xc,2) + pow(x[1]-yc,2)) - R) / eps ) ",
    degree=4,  # polynomial degree used for interpolation
    xc=0.5,
    yc=0.75,
    R=0.15,
    eps=eps
)

c_n = fe.project(c_init_expr, V)

F = c_nP1*v*fe.dx + dt*fe.dot(vel_n, fe.grad(c_nP1))*v*fe.dx\
    + dt*mobility(c_n)*fe.dot(fe.grad(c_nP1),fe.grad(v))*fe.dx - c_n*v*fe.dx 
    
    

t = 0
for n in range(num_steps):
    
    t += dt
    u_expr.t = t
    vel_n.assign(fe.project(u_expr, V_vec))
    
    fe.solve(F == 0, c_nP1)
    
    c_n.assign(c_nP1)
    
    if n % 100 == 0:  # plot every 10 steps
        coords = mesh.coordinates()
        c_vals = c_n.compute_vertex_values(mesh)
        triangles = mesh.cells()  # get mesh connectivity
        triang = tri.Triangulation(coords[:, 0], coords[:, 1], triangles)
    
        plt.figure(figsize=(6,5))
        plt.tricontourf(triang, c_vals, levels=50, cmap="RdBu_r")
        plt.colorbar(label=r"$\phi$")
        plt.title(f"phi at t = {t:.2f}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()
        
        # Save the figure to your output folder
        out_file = os.path.join(outDirName, f"phi_t{n:05d}.png")
        plt.savefig(out_file, dpi=200)
        plt.show()



