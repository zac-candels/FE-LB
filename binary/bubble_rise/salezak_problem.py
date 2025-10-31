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
outDirName = os.path.join(WORKDIR, "figures")
os.makedirs(outDirName, exist_ok=True)

L_x, L_y = 100, 100
nx, ny = 100, 100
h = L_x / nx
dt = h / 4. 

rad = 15
circ_center_x = 50
circ_center_y = 75
notch_height = 25
notch_size = 6

# Set up domain. For simplicity, do unit square mesh.

mesh = fe.RectangleMesh(fe.Point(0, 0), fe.Point(L_x, L_y), nx, nx)

# Set periodic boundary conditions at left and right endpoints


class PeriodicBoundaryY(fe.SubDomain):
    def inside(self, point, on_boundary):
        return fe.near(point[1], 0.0) and on_boundary

    def map(self, top_bdy, bottom_bdy):
        # Correct mapping for periodicity in Y (mapping y=Ly to y=0)
        bottom_bdy[0] = top_bdy[0]  # X-coordinate remains the same
        bottom_bdy[1] = top_bdy[1] - L_y # Y-coordinate shifts by -L_y


pbc = PeriodicBoundaryY()


V = fe.FunctionSpace(mesh, "P", 1, constrained_domain=pbc)


# Define trial and test functions, as well as
# finite element functions at previous timesteps

u_trial = fe.TrialFunction(V)
phi_n = fe.Function(V)
V_vec = fe.VectorFunctionSpace(mesh, "P", 1, constrained_domain=pbc)
vel_n = fe.Function(V_vec)
mu_n = fe.Function(V)

v = fe.TestFunction(V)

phi_nP1 = fe.Function(V)




# Define Allen-Cahn mobility

def mobility(phi_n):
    grad_phi_n = fe.grad(phi_n)
    
    abs_grad_phi_n = fe.sqrt(fe.dot(grad_phi_n, grad_phi_n) + 1e-12)
    inv_abs_grad_phi_n = 1.0 / abs_grad_phi_n
    
    mob = M_tilde*( 1 - 4*phi_n*(1 - phi_n)/eps * inv_abs_grad_phi_n )
    return mob
    

# Initialize \phi
phi_init = phi_init_expr = fe.Expression(
    "0.5 + 0.5 * tanh( 2.0 * ( sqrt( pow(x[0]-xc,2) + pow(x[1]-yc,2) ) - R) / eps )",
    degree=2,  # polynomial degree used for interpolation
    xc=center_init_x,
    yc=center_init_y,
    R=initBubbleDiam/2,
    eps=eps
)

phi_n = fe.interpolate(phi_init_expr, V)

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


tol = 1e-8


# Define variational problems


n = fe.FacetNormal(mesh)

bilin_form = u_trial * v * fe.dx

lin_form_AC = phi_n * v * fe.dx - dt*v*fe.dot(vel_n, fe.grad(phi_n))*fe.dx\
    - dt*fe.dot(fe.grad(v), mobility(phi_n)*fe.grad(phi_n))*fe.dx\
        - 0.5*dt**2 * fe.dot(vel_n, fe.grad(v)) * fe.dot(vel_n, fe.grad(phi_n)) *fe.dx\
           

lin_form_mu = 4*beta*(phi_n - 1)*(phi_n - 0)*(phi_n - 0.5)*v*fe.dx\
    + kappa*fe.dot(fe.grad(phi_n),fe.grad(v))*fe.dx 


sys_mat = fe.assemble(bilin_form)

# Assemble matrices for first step

rhs_mu = fe.assemble(lin_form_mu)
fe.solve(sys_mat, mu_n.vector(), rhs_mu)

# Timestepping
t = 0.0
for n in range(num_steps):
    t += dt
    
    rhs_AC = fe.assemble(lin_form_AC)
    rhs_mu = fe.assemble(lin_form_mu)
    
        
    
    fe.solve(sys_mat, phi_nP1.vector(), rhs_AC)
    fe.solve(sys_mat, mu_n.vector(), rhs_mu)
