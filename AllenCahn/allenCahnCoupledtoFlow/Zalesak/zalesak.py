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

T = 1500
CFL = 0.2
L_x, L_y = 100, 100
nx, ny = 200, 400
h = L_x/nx
dt = h/4
num_steps = int(np.ceil(T/dt))


sigma = 0.05
eps = 0.7*h

beta = 12*sigma/eps
kappa = 3*sigma*eps/2

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

v = fe.TestFunction(V)

phi_nP1 = fe.Function(V)



# Define Allen-Cahn mobility
def mobility(phi_n):
    grad_phi_n = fe.grad(phi_n)

    abs_grad_phi_n = fe.sqrt(fe.dot(grad_phi_n, grad_phi_n) + 1e-6)
    inv_abs_grad_phi_n = 1.0 / abs_grad_phi_n

    mob = 0.5*(50*np.pi/314)*( 1 - 4*phi_n*(1 - phi_n)/eps * inv_abs_grad_phi_n )
    return mob


# ----------------------------
# 1) Initialize phi as Zalesak's notched disk (constructed on mesh coords, then interpolated)
# ----------------------------
# Notched-disk parameters (you can tweak these)
R = 15          # disk radius
notch_width = 2                 # notch width (in y-direction)
notch_depth = 25                # how far the notch cuts into the disk (in x-direction)
xc = center_init_x
yc = center_init_y

# Evaluate a signed-distance-like field on mesh vertices and make smooth tanh profile
coords = mesh.coordinates()
phi_vals = np.zeros(coords.shape[0])

# compute signed distance to circle (positive outside, negative inside)
dist = np.sqrt((coords[:,0]-xc)**2 + (coords[:,1]-yc)**2) - R

# identify notch rectangle (this notch sits to the left of the disk center)
# We'll create a notch whose rectangular region removes the disk where:
# x in [xc - R, xc - R + notch_depth] and |y - yc| < notch_width/2
x_cond = (coords[:,0] >= (xc - R)) & (coords[:,0] <= (xc - R + notch_depth))
y_cond = np.abs(coords[:,1] - yc) <= (notch_width/2.0)
notch_mask = x_cond & y_cond

# For points within the notch rectangle we push them well outside the disk (make dist large positive)
# to remove them from the 'inside' set.
dist[notch_mask] = np.maximum(dist[notch_mask], 2.0*R)

# Now create a smooth phi via a tanh interface (same structure you used before)
# choose interface sharpness factor (consistent with your earlier expression)
phi_vals = 0.5 + 0.5 * np.tanh(-2.0 * dist / eps)  # negative dist => inside => phi~1

# Create a Function and assign node values (do a direct interpolation)
phi_init = fe.Function(V)
# assign values at vertex DOFs: map mesh coordinates to dof ordering
# For P1 scalar on triangles, tabulate_dof_coordinates returns coords in dof order
dof_coords = V.tabulate_dof_coordinates().reshape((-1, 2))
# Build a KD-tree-like mapping by matching physical coordinates (floating equality is OK since same mesh)
# We'll use a simple nearest neighbor mapping (should be exact because coords arrays come from same mesh)
from scipy.spatial import cKDTree
tree = cKDTree(coords)
_, idx = tree.query(dof_coords, k=1)
phi_init.vector().set_local(phi_vals[idx])
phi_init.vector().apply("insert")

phi_n.assign(phi_init)

# Plot initial phi
coords_plot = mesh.coordinates()
phi_plot_vals = phi_n.compute_vertex_values(mesh)
triangles = mesh.cells()  # get mesh connectivity
triang = tri.Triangulation(coords_plot[:, 0], coords_plot[:, 1], triangles)

plt.figure(figsize=(6,5))
plt.tricontourf(triang, phi_plot_vals, levels=90, cmap="RdBu_r")
plt.colorbar(label=r"$\phi$")
plt.title(f"phi at t = {0:.2f}")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()

# Save the figure to your output folder
out_file = os.path.join(outDirName, f"phi_t{0:05d}.png")
plt.savefig(out_file, dpi=200)
plt.show()


# ----------------------------
# 2) Add Zalesak velocity field and project into V_vec
#    u = pi*(50 - y)/314, v = pi*(x - 50)/314
# ----------------------------
# You asked to include exactly this velocity. We create it as an Expression and project it.
u_expr = fe.Expression(
    ("pi*(yc - x[1])/314.0", "pi*(x[0] - xc)/314.0"),
    degree=2,
    pi=np.pi,
    xc=center_init_x,
    yc=center_init_y
)

# Project onto vector finite element space (periodic in y via same constrained_domain)
vel_n = fe.project(u_expr, V_vec)

# (Optionally) you can visualize velocity early on; below we visualize at the end as in your original script.

# ----------------------------
# Weak forms (unchanged, but now vel_n contains the Zalesak velocity)
# ----------------------------
bilin_form_AC = f_trial * v * fe.dx

lin_form_AC = phi_n * v * fe.dx - dt*v*fe.dot(vel_n, fe.grad(phi_n))*fe.dx\
    - dt*fe.dot(fe.grad(v), mobility(phi_n)*fe.grad(phi_n))*fe.dx\
        - 0.5*dt**2 * fe.dot(vel_n, fe.grad(v)) * fe.dot(vel_n, fe.grad(phi_n)) *fe.dx\


lin_form_mu = 4*beta*(phi_n - 1)*(phi_n - 0)*(phi_n - 0.5)*v*fe.dx\
    + kappa*fe.dot(fe.grad(phi_n),fe.grad(v))*fe.dx


phi_mat = fe.assemble(bilin_form_AC)
mu_mat = fe.assemble(bilin_form_AC)
rhs_mu = fe.assemble(lin_form_mu)

fe.solve(mu_mat, mu_n.vector(), rhs_mu)

# Timestepping
t = 0.0
for n in range(num_steps):
    t += dt

    # if you want the velocity to change in time you could re-project here; for Zalesak it's steady
    # vel_n = fe.project(u_expr, V_vec)

    rhs_AC = fe.assemble(lin_form_AC)
    rhs_mu = fe.assemble(lin_form_mu)

    fe.solve(phi_mat, phi_nP1.vector(), rhs_AC)
    fe.solve(mu_mat, mu_n.vector(), rhs_mu)

    # Update previous solutions
    phi_n.assign(phi_nP1)

    if n % 5 == 0:
        coords_plot = mesh.coordinates()
        phi_plot_vals = phi_n.compute_vertex_values(mesh)
        triangles = mesh.cells()  # get mesh connectivity
        triang = tri.Triangulation(coords_plot[:, 0], coords_plot[:, 1], triangles)

        print("t = ", t)
        plt.figure(figsize=(6,5))
        plt.tricontourf(triang, phi_plot_vals, levels=50, cmap="RdBu_r")
        plt.colorbar(label=r"$\phi$")
        plt.title(f"phi at t = {t:.6f}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()

        # Save the figure to your output folder
        ##plt.close()
        
        a  =1


#%%
u = vel_n
# Plot velocity field with larger arrows
coords = V_vec.tabulate_dof_coordinates()[::2]
u_values = u.vector().get_local().reshape((V_vec.dim() // 2, 2))
x = coords[:, 0]
y = coords[:, 1]
u_x = u_values[:, 0]
u_y = u_values[:, 1]

# Define arrow scale based on maximum velocity
max_u = np.max(np.sqrt(u_x**2 + u_y**2))
arrow_length = 0.05  # 5% of domain size
scale = max_u / arrow_length if max_u > 0 else 1

plt.figure()
M = np.hypot(u_x, u_y)
plt.quiver(x, y, u_x, u_y, M, scale=scale, scale_units='height')
plt.title("Velocity field at final time")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# %% Create grid of u_x and u_y values
coords = V_vec.tabulate_dof_coordinates()[::2]  # Shape: (1056, 2)
u_values = vel_n.vector().get_local().reshape(
    (V_vec.dim() // 2, 2))  # Shape: (1056, 2)
x = coords[:, 0]  # x-coordinates
y = coords[:, 1] 
# figure out unique x- and y- levels
x_unique = np.unique(x)
y_unique = np.unique(y)
num_x_unique = len(x_unique)
num_y_unique = len(y_unique)
assert num_x_unique*num_y_unique == u_x.size, "grid size mismatch"

# now sort the flat arrays into lexicographic (y,x) order
order = np.lexsort((x, y))

# apply that ordering
u_x_sorted = u_x[order]
u_y_sorted = u_y[order]

# reshape into (ny, nx).  If your mesh is square, nx==ny.
u_x_grid = u_x_sorted.reshape((num_y_unique, num_x_unique))
u_y_grid = u_y_sorted.reshape((num_y_unique, num_x_unique))
