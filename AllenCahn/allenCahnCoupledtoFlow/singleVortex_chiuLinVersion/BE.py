import fenics as fe
import os
import numpy as np
#import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri

plt.close('all')

fe.parameters["std_out_all_processes"] = False
fe.set_log_level(fe.LogLevel.ERROR)

# Where to save the plots
WORKDIR = os.getcwd()
outDirName = os.path.join(WORKDIR, "singleVortexBE")
os.makedirs(outDirName, exist_ok=True)

T = 5
CFL = 0.2
L_x, L_y = 1, 1
nx, ny = 128, 128
h = L_x/nx
dt = h**2
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

    mob = c_n*(1-c_n) * inv_abs_grad_c_n
    return mob


# Initialize \c

tol = eps

class CircleExpr(fe.UserExpression):
    def __init__(self, xc, yc, R, tol, inside=True, **kwargs):
        super().__init__(**kwargs)
        self.xc = xc
        self.yc = yc
        self.R  = R
        self.tol = tol
        self.inside = inside
    def eval(self, values, x):
        dx = x[0] - self.xc
        dy = x[1] - self.yc
        d = (dx*dx + dy*dy)**0.5
        if self.inside:
            # 1 inside disk, 0 outside
            values[0] = 1.0 if d <= self.R + 1e-12 else 0.0
        else:
            # 1 on the circle boundary within tol, 0 elsewhere
            values[0] = 1.0 if abs(d - self.R) <= self.tol else 0.0
    def value_shape(self):
        return ()

c_init_expr = CircleExpr(xc, yc, R, tol, inside=True, degree=2)

c_n = fe.interpolate(c_init_expr, V)

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

F = c_nP1*v*fe.dx - dt*c_nP1*fe.dot( vel_n, fe.grad(v) )*fe.dx\
    + dt*gammaBar*eps*fe.dot( fe.grad(c_nP1), fe.grad(v) )*fe.dx\
    - dt*mobility(c_n)*fe.dot(fe.grad(c_nP1),fe.grad(v))*fe.dx - c_n*v*fe.dx 
    
    

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
        #plt.show()
        
        a=1



