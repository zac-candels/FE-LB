import fenics as fe
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import time
 
comm = fe.MPI.comm_world
rank = fe.MPI.rank(comm)

start_time = time.time() 

plt.close('all')

# Where to save the plots


T = 1500
CFL = 0.2
initBubbleDiam = 5
L_x, L_y = 2*initBubbleDiam, 2*initBubbleDiam
nx, ny = 100, 100
h = min(L_x/nx, L_y/ny)
dt = h*CFL
num_steps = int(np.ceil(T/dt))


g = 0.0981
sigma = 0.005 #0.1

# Cahn number
Cn = 0.05

eps = Cn * initBubbleDiam

beta = 12*sigma/eps

kappa = 3*sigma*eps/2 

# Relaxation times for heavier and lighter phases
tau_h = 1 #eta_h / (c_s2 * rho_h * dt )
tau_l = 1# eta_l / (c_s2 * rho_l * dt )

theta_deg = 30
theta = theta_deg * np.pi / 180

WORKDIR = os.getcwd()
outDirName = os.path.join(WORKDIR, f"1D")
os.makedirs(outDirName, exist_ok=True)

M_tilde = 0.01

center_init_x, center_init_y = L_x/2, initBubbleDiam/2 - 2

# Set up domain. For simplicity, do unit square mesh.

mesh = fe.RectangleMesh(comm, fe.Point(0, 0), fe.Point(L_x, L_y), nx, ny, diagonal="crossed")

# Set periodic boundary conditions at left and right endpoints


class PeriodicBoundaryX(fe.SubDomain):
    def inside(self, point, on_boundary):
        return fe.near(point[0], 0.0) and on_boundary

    def map(self, right_bdy, left_bdy):
        # Map left boundary to the right
        left_bdy[0] = right_bdy[0] - L_x
        left_bdy[1] = right_bdy[1]


pbc = PeriodicBoundaryX()


V = fe.FunctionSpace(mesh, "P", 1, constrained_domain=pbc)


# Define trial and test functions, as well as
# finite element functions at previous timesteps

f_trial = fe.TrialFunction(V)
f_n = []
for idx in range(Q):
    f_n.append(fe.Function(V))
phi_n = fe.Function(V)
V_vec = fe.VectorFunctionSpace(mesh, "P", 1, constrained_domain=pbc)
vel_n = fe.Function(V_vec)
mu_n = fe.Function(V)

v = fe.TestFunction(V)



# Define Allen-Cahn mobility

def mobility(phi_n):
    grad_phi_n = fe.grad(phi_n)
    
    abs_grad_phi_n = fe.sqrt(fe.dot(grad_phi_n, grad_phi_n) + 1e-6)
    inv_abs_grad_phi_n = 1.0 / abs_grad_phi_n
    
    mob = M_tilde*( 1 - 4*phi_n*(1 - phi_n)/eps * inv_abs_grad_phi_n )
    return mob
    


# # Initialize distribution functions. We will use
# where \bar{u}_0 = u_0 - F\Delta t/( 2 \rho_0 ).
# Here we will take u_0 = 0.

for idx in range(Q):
    f_n[idx] = (fe.project(f_equil_init(idx), V))
    
# Initialize \phi
phi_init_expr = fe.Expression(
    "0.5 - 0.5 * tanh( 2.0 * (sqrt(pow(x[0]-xc,2) + pow(x[1]-yc,2)) - R) / eps )",
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


n = fe.FacetNormal(mesh)
opp_idx = {0: 0, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}


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
bilin_form_mu = f_trial * v * fe.dx

lin_form_AC = phi_n * v * fe.dx - dt*v*fe.dot(vel_n, fe.grad(phi_n))*fe.dx\
    - dt*fe.dot(fe.grad(v), mobility(phi_n)*fe.grad(phi_n))*fe.dx\
        - 0.5*dt**2 * fe.dot(vel_n, fe.grad(v)) * fe.dot(vel_n, fe.grad(phi_n)) *fe.dx\
            - dt*np.cos(theta)*v*mobility(phi_n)*4*phi_n*(1 - phi_n)/eps*ds_bottom

lin_form_mu = 4*beta*(phi_n - 1)*(phi_n - 0)*(phi_n - 0.5)*v*fe.dx\
    + kappa*fe.dot(fe.grad(phi_n),fe.grad(v))*fe.dx\
        #- (1/kappa)*np.cos(theta)*np.sqrt(2*kappa*beta)*v*(phi_n - phi_n**2)*ds_bottom


# Assemble matrices for first step

rhs_vec_streaming = [0]*Q
rhs_vec_collision = [0]*Q

sys_mat = []
for idx in range(Q):
    sys_mat.append(fe.assemble(bilinear_forms_stream[idx]))
    
solver_list = []
for idx in range(Q):
    A = sys_mat[idx]

    # Create CG solver
    solver = fe.KrylovSolver("cg", "hypre_amg")  # use ILU preconditioner
    solver.set_operator(A)

    # Optional: set solver parameters
    prm = solver.parameters
    prm["absolute_tolerance"] = 1e-12
    prm["relative_tolerance"] = 1e-8
    prm["maximum_iterations"] = 1000
    prm["nonzero_initial_guess"] = False

    solver_list.append(solver)

phi_mat = fe.assemble(bilin_form_AC)
mu_mat = fe.assemble(bilin_form_mu)
phi_solver = fe.LUSolver("mumps")
phi_solver.set_operator(phi_mat)

mu_solver = fe.LUSolver("mumps")
mu_solver.set_operator(mu_mat)


rhs_mu = fe.assemble(lin_form_mu)

fe.solve(mu_mat, mu_n.vector(), rhs_mu)

outfile = fe.XDMFFile(comm, f"{outDirName}/solution.xdmf")
outfile.parameters["flush_output"] = True
outfile.parameters["functions_share_mesh"] = True
outfile.parameters["rewrite_function_mesh"] = False

# Timestepping
t = 0.0
for n in range(num_steps):
    t += dt
    
    print("n = ", n)
    
    rhs_AC = fe.assemble(lin_form_AC)
    
    # Solve linear system in each timestep, get f^{n+1}
    for idx in range(Q):
        solver_list[idx].solve(f_nP1[idx].vector(), rhs_vec_streaming[idx])
        
    
    phi_solver.solve(phi_nP1.vector(), rhs_AC)
    mu_solver.solve(mu_n.vector(), rhs_mu)


    # Update previous solutions

    for idx in range(Q):
        f_n[idx].assign(f_nP1[idx])
    phi_n.assign(phi_nP1)
    vel_expr = vel(f_n)
    fe.project(vel_expr, V_vec, function=vel_n)
    
    if rank == 0:
        if n % 10 == 0:  # plot every 10 steps
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
            #plt.show()
            plt.close()
                


# %%
u_expr = vel(f_n)
V_vec = fe.VectorFunctionSpace(mesh, "P", 1, constrained_domain=pbc)
u = fe.project(u_expr, V_vec)


# Plot velocity field with larger arrows
# Plot velocity field with larger arrows
coords = V_vec.tabulate_dof_coordinates()[::2]  # Shape: (1056, 2)
u_values = u.vector().get_local().reshape(
    (V_vec.dim() // 2, 2))  # Shape: (1056, 2)
x = coords[:, 0]  # x-coordinates
y = coords[:, 1]  # y-coordinates
u_x = u_values[:, 0]  # x-components of velocity
u_y = u_values[:, 1]  # y-components of velocity

# Define arrow scale based on maximum velocity
max_u = np.max(np.sqrt(u_x**2 + u_y**2))
arrow_length = 0.05  # 5% of domain size
scale = max_u / arrow_length if max_u > 0 else 1

# Create quiver plot
plt.figure()
M = np.hypot(u_x, u_y)
plt.quiver(x, y, u_x, u_y, M, scale=scale, scale_units='height')
plt.title("Velocity field at final time")
plt.xlabel("x")
plt.ylabel("y")
#plt.show()






# %% Create grid of u_x and u_y values

# figure out unique x- and y- levels
x_unique = np.unique(x)
y_unique = np.unique(y)
num_x_unique = len(x_unique)
num_y_unique = len(y_unique)
assert num_x_unique*num_y_unique == u_x.size, "grid size mismatch"

# now sort the flat arrays into lexicographic (y,x) order
# we want the slow index to be y, fast index x, so lexsort on (x,y)
order = np.lexsort((x, y))

# apply that ordering
u_x_sorted = u_x[order]
u_y_sorted = u_y[order]

# reshape into (ny, nx).  If your mesh is square, nx==ny.
u_x_grid = u_x_sorted.reshape((num_y_unique, num_x_unique))
u_y_grid = u_y_sorted.reshape((num_y_unique, num_x_unique))


# %% Create 2D grids of each f_i at final time

# 1) Extract the coordinates of each degree of freedom in V
coords_f = V.tabulate_dof_coordinates().reshape(-1, 2)
x_f = coords_f[:, 0]
y_f = coords_f[:, 1]

# 2) Find unique levels and check grid size
x_unique = np.unique(x_f)
y_unique = np.unique(y_f)
nx_f = len(x_unique)
ny_f = len(y_unique)
assert nx_f * ny_f == x_f.size, "grid size mismatch for f_i"

# 3) Compute lexicographic ordering so that slow index=y, fast=x
order_f = np.lexsort((x_f, y_f))

# 4) Loop over all distributions, sort & reshape
f_grids = []
for idx, fi in enumerate(f_n):
    # flatten values, sort into (y,x) lex order, then reshape into (ny, nx)
    fi_vals = fi.vector().get_local()
    fi_sorted = fi_vals[order_f]
    fi_grid = fi_sorted.reshape((ny_f, nx_f))
    f_grids.append(fi_grid)
    # Optional: if you want to name them individually:
    # globals()[f"f{idx}_grid"] = fi_grid

# Now f_grids[i] is the (ny_f × nx_f) array of f_i values at the mesh grid.
# e.g., f_grids[0] is f0_grid, f_grids[1] is f1_grid, etc.

#%% Create grid for \phi at final time

# 1) Extract the coordinates of each degree of freedom in V
coords = V.tabulate_dof_coordinates().reshape(-1, 2)
x = coords[:, 0]
y = coords[:, 1]

# 2) Find unique levels and check grid size
x_unique = np.unique(x)
y_unique = np.unique(y)
nx = len(x_unique)
ny = len(y_unique)
assert nx * ny == x.size, "grid size mismatch for f_i"

# 3) Compute lexicographic ordering so that slow index=y, fast=x
order_phi = np.lexsort((x, y))

# 4) Loop over all distributions, sort & reshape
phi_grid = []
# flatten values, sort into (y,x) lex order, then reshape into (ny, nx)
phi_vals = phi_n.vector().get_local()
phi_sorted = phi_vals[order_phi]
phi_grid = phi_sorted.reshape((ny, nx))
    # Optional: if you want to name them individually:
    # globals()[f"f{idx}_grid"] = fi_grid

end_time = time.time()

print("time elapsed: ", end_time - start_time)