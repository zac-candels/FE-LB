import fenics as fe
import os
import numpy as np
#import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri

plt.close('all')

# Where to save the plots
WORKDIR = os.getcwd()
outDirName = os.path.join(WORKDIR, "zalesakFigures")
os.makedirs(outDirName, exist_ok=True)

T = 4
CFL = 0.2
L_x, L_y = 1, 1
nx, ny = 128, 128
h = L_x/nx
dt = h/480
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

    mob = (1/eps)*( 1 - 4*phi_n*(1 - phi_n)/eps * inv_abs_grad_phi_n )
    return mob


# Initialize \phi
phi_init_expr = fe.Expression(
    "0.5 - 0.5 * tanh( 2.0 * (sqrt(pow(x[0]-xc,2) + pow(x[1]-yc,2)) - R) / eps ) ",
    degree=4,  # polynomial degree used for interpolation
    xc=0.5,
    yc=0.75,
    R=0.15,
    eps=eps
)

phi_n = fe.project(phi_init_expr, V)

coords = mesh.coordinates()
phi_vals = phi_n.compute_vertex_values(mesh)
triangles = mesh.cells()  # get mesh connectivity
triang = tri.Triangulation(coords[:, 0], coords[:, 1], triangles)

plt.figure(figsize=(6,5))
plt.tricontourf(triang, phi_vals, levels=50, cmap="RdBu_r")
plt.colorbar(label=r"$\phi$")
plt.title(f"phi at t = {0:.2f}")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()

# Save the figure to your output folder
out_file = os.path.join(outDirName, f"phi_t{0:05d}.png")
plt.savefig(out_file, dpi=200)
plt.show()
#plt.close()



# Define boundary conditions.

# For f_5, f_2, and f_6, equilibrium boundary conditions at lower wall
# Since we are applying equilibrium boundary conditions
# and assuming no slip on solid walls, f_i^{eq} reduces to
# \rho * w_i




# Create MeshFunction for boundary markers
boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)

# Subdomain for bottom wall
class Bottom(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and fe.near(x[1], 0.0)

bottom = Bottom()
bottom.mark(boundaries, 1)   # assign ID = 1 to bottom boundary
ds_bottom = fe.Measure("ds", domain=mesh, subdomain_data=boundaries, subdomain_id=1)

bilin_form_AC = f_trial * v * fe.dx

lin_form_AC = phi_n * v * fe.dx - dt*v*fe.dot(vel_n, fe.grad(phi_n))*fe.dx\
    - dt*fe.dot(fe.grad(v), mobility(phi_n)*fe.grad(phi_n))*fe.dx\
        - 0.5*dt**2 * fe.dot(vel_n, fe.grad(v)) * fe.dot(vel_n, fe.grad(phi_n)) *fe.dx\
           


phi_mat = fe.assemble(bilin_form_AC)
phi_solver = fe.KrylovSolver("cg", "ilu")
phi_solver.set_operator(phi_mat)


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
    
    if n % 1 == 0:  # plot every 10 steps
        coords = mesh.coordinates()
        phi_vals = phi_n.compute_vertex_values(mesh)
        triangles = mesh.cells()  # get mesh connectivity
        triang = tri.Triangulation(coords[:, 0], coords[:, 1], triangles)
    
        plt.figure(figsize=(6,5))
        plt.tricontourf(triang, phi_vals, levels=50, cmap="RdBu_r")
        plt.colorbar(label=r"$\phi$")
        plt.title(f"phi at t = {t:.2f}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()
        
        # Save the figure to your output folder
        out_file = os.path.join(outDirName, f"phi_t{n:05d}.png")
        plt.savefig(out_file, dpi=200)
        plt.show()
        
        a = 1
                


# %%





