import fenics as fe
import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri

plt.close('all')

# Where to save the plots
WORKDIR = os.getcwd()
outDirName = os.path.join(WORKDIR, "figures")
os.makedirs(outDirName, exist_ok=True)

T = 1500
CFL = 0.2
initBubbleDiam = 5
L_x, L_y = 4*initBubbleDiam, 4*initBubbleDiam
nx, ny = 200, 400
h = L_x/nx
dt = h*CFL
num_steps = int(np.ceil(T/dt))


g = 0.0981
sigma = 0.005 #0.1

# Lattice speed of sound
c_s = np.sqrt(1/3)
c_s2 = 1/3

# Bond number
Bo = 100

# Morton number
Mo = 1000

# Cahn number
Cn = 0.05

eps = Cn * initBubbleDiam

rho_h = 1#sigma*Bo*g / initBubbleDiam**2
rho_l = 1#rho_h/100

eta_h = (Mo * sigma**3 * rho_h)**(1/4)
eta_l = eta_h/100

beta = 12*sigma/eps

kappa = 3*sigma*eps/2 

# Relaxation times for heavier and lighter phases
tau_h = 1 #eta_h / (c_s2 * rho_h * dt )
tau_l = 1# eta_l / (c_s2 * rho_l * dt )

theta = 30 * np.pi / 180

M_tilde = 0.01

center_init_x, center_init_y = L_x/2, initBubbleDiam/2 - 2

Q = 9
# D2Q9 lattice velocities
xi = [
    fe.Constant((0.0,  0.0)),
    fe.Constant((1.0,  0.0)),
    fe.Constant((0.0,  1.0)),
    fe.Constant((-1.0,  0.0)),
    fe.Constant((0.0, -1.0)),
    fe.Constant((1.0,  1.0)),
    fe.Constant((-1.0,  1.0)),
    fe.Constant((-1.0, -1.0)),
    fe.Constant((1.0, -1.0)),
]

# Corresponding weights
w = np.array([
    4/9,
    1/9, 1/9, 1/9, 1/9,
    1/36, 1/36, 1/36, 1/36
])

# Set up domain. For simplicity, do unit square mesh.

mesh = fe.RectangleMesh(fe.Point(0, 0), fe.Point(L_x, L_y), nx, nx)

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

# Define FE functions to hold post-streaming solution at nP1 timesteps
f_nP1 = []
for idx in range(Q):
    f_nP1.append(fe.Function(V))
phi_nP1 = fe.Function(V)

# Define FE functions to hold post-collision distributions
f_star = []
for idx in range(Q):
    f_star.append(fe.Function(V))


# Define density
def rho(phi):
    return rho_h*phi + rho_l*(1 - phi)

def tau_fn(phi):
    return phi*tau_h + (1 - phi)*tau_l

# Define dynamic pressure
def get_pBar(f_list):
    return f_list[0] + f_list[1] + f_list[2] + f_list[3] + f_list[4]\
        + f_list[5] + f_list[6] + f_list[7] + f_list[8]

# Define velocity

def vel(f_list):
    distr_fn_sum = f_list[0]*xi[0] + f_list[1]*xi[1] + f_list[2]*xi[2]\
        + f_list[3]*xi[3] + f_list[4]*xi[4] + f_list[5]*xi[5]\
        + f_list[6]*xi[6] + f_list[7]*xi[7] + f_list[8]*xi[8]

    velocity = distr_fn_sum/c_s**2

    return velocity


# Define initial equilibrium distributions
def f_equil_init(vel_idx):
    
    # We'll take \bar{p} := 1.0
    return w[vel_idx] 


# Define equilibrium distribution
# def f_equil(f_list, vel_idx):
#     dyn_pres_expr = dyn_pres(f_list)
#     u_expr = vel(f_list)
#     ci = xi[vel_idx]
#     ci_dot_u = fe.dot(ci, u_expr)
#     return w[vel_idx] * (
#         dyn_pres_expr
#         + ci_dot_u
#         + ci_dot_u**2 / (2*c_s**2)
#         - fe.dot(u_expr, u_expr) / 2
#     )

xi_array = np.array([[float(c.values()[0]), float(c.values()[1])] for c in xi])

def f_equil(f_list, idx):
    """
    Compute equilibrium distribution for direction idx
    Returns a NumPy array (values at all DoFs).
    """
    # Number of DoFs
    N = f_list[0].vector().size()

    # Stack all f_i values: shape (Q, N)
    f_stack = np.array([f.vector().get_local() for f in f_list])

    # Compute density at each DoF
    p_bar = np.sum(f_stack, axis=0)  # shape (N,)

    # Compute velocity at each DoF
    ux_vec = np.sum(f_stack * xi_array[:,0][:,None], axis=0) / c_s**2
    uy_vec = np.sum(f_stack * xi_array[:,1][:,None], axis=0) / c_s**2

    u2 = ux_vec**2 + uy_vec**2

    # Compute ci . u for this direction
    cu = xi_array[idx,0]*ux_vec + xi_array[idx,1]*uy_vec
    
    feq = w[idx]*( p_bar + cu + cu**2/(2*c_s**2) - u2/2)

    return feq  # NumPy array


# Define \Gamma
def Gamma_vel(vel_arg, vel_idx):
    ci = xi[vel_idx]
    ci_dot_u = fe.dot(ci, vel_arg)
    return w[vel_idx] * (
        1
        + ci_dot_u / (c_s**2)
        + ci_dot_u**2 / (2*c_s**4)
        - fe.dot(vel_arg, vel_arg) / (2*c_s**2)
    )    
    
def Gamma0(vel_idx):
    return w[vel_idx]
    

def body_Force(f_list, phi, mu, vel_idx):
    
    grav = fe.Constant((0.0, 9.81))
    p_bar = get_pBar(f_list)
    fluid_vel = vel(f_list)
    density = rho(phi)
    p = density * p_bar 
    buoyancy = - grav*(density - rho_h)
    tau = tau_fn(phi)
    
    eta = c_s**2 * density * tau * dt
    
    p_bar_grad = fe.grad(p_bar)
    p_grad = fe.grad(p)
    phi_grad = fe.grad(phi)
    density_grad = fe.grad(density)
    
    sym_grad_u = 2 * fe.sym(fe.grad(fluid_vel))
    
    term1 = -Gamma_vel(fluid_vel, vel_idx)\
        *fe.dot( (xi[vel_idx] - fluid_vel),  p_grad/density ) 
    
    term2 = Gamma0(vel_idx)*fe.dot( (xi[vel_idx] - fluid_vel), p_bar_grad )
    
    term3 = Gamma_vel(fluid_vel, vel_idx)\
        *fe.dot( (xi[vel_idx] - fluid_vel), phi_grad*mu/density )
        
    term4 = Gamma_vel(fluid_vel, vel_idx)\
        *fe.dot( (xi[vel_idx] - fluid_vel),  sym_grad_u* density_grad )\
            * (eta/density**2)
    

    return term1 + term2 + term3 + term4


# Define Allen-Cahn mobility

def mobility(phi_n):
    grad_phi_n = fe.grad(phi_n)
    
    abs_grad_phi_n = fe.sqrt(fe.dot(grad_phi_n, grad_phi_n) + 1e-12)
    inv_abs_grad_phi_n = 1.0 / abs_grad_phi_n
    
    mob = M_tilde*( 1 - 4*phi_n*(1 - phi_n)/eps * inv_abs_grad_phi_n )
    return mob
    


# # Initialize distribution functions. We will use
# where \bar{u}_0 = u_0 - F\Delta t/( 2 \rho_0 ).
# Here we will take u_0 = 0.

for idx in range(Q):
    f_n[idx] = (fe.project(f_equil_init(idx), V))
    
# Initialize \phi
phi_init = phi_init_expr = fe.Expression(
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



# Define boundary conditions.

# For f_5, f_2, and f_6, equilibrium boundary conditions at lower wall
# Since we are applying equilibrium boundary conditions
# and assuming no slip on solid walls, f_i^{eq} reduces to
# \rho * w_i

tol = 1e-8


def Bdy_Lower(x, on_boundary):
    if on_boundary:
        if fe.near(x[1], 0, tol):
            return True
        else:
            return False
    else:
        return False


rho_expr = sum(fk for fk in f_n)

f5_lower = f_n[7]  # rho_expr
f2_lower = f_n[4]  # rho_expr
f6_lower = f_n[8]  # rho_expr

f5_lower_func = fe.Function(V)
f2_lower_func = fe.Function(V)
f6_lower_func = fe.Function(V)

fe.project(f5_lower, V, function=f5_lower_func)
fe.project(f2_lower, V, function=f2_lower_func)
fe.project(f6_lower, V, function=f6_lower_func)

bc_f5 = fe.DirichletBC(V, f5_lower_func, Bdy_Lower)
bc_f2 = fe.DirichletBC(V, f2_lower_func, Bdy_Lower)
bc_f6 = fe.DirichletBC(V, f6_lower_func, Bdy_Lower)

# Similarly, we will define boundary conditions for f_7, f_4, and f_8
# at the upper wall. Once again, boundary conditions simply reduce
# to \rho * w_i


tol = 1e-8


def Bdy_Upper(x, on_boundary):
    if on_boundary:
        if fe.near(x[1], L_y, tol):
            return True
        else:
            return False
    else:
        return False


rho_expr = sum(fk for fk in f_n)

f7_upper = f_n[5]  # rho_expr
f4_upper = f_n[2]  # rho_expr
f8_upper = f_n[6]  # rho_expr

f7_upper_func = fe.Function(V)
f4_upper_func = fe.Function(V)
f8_upper_func = fe.Function(V)

fe.project(f7_upper, V, function=f7_upper_func)
fe.project(f4_upper, V, function=f4_upper_func)
fe.project(f8_upper, V, function=f8_upper_func)

bc_f7 = fe.DirichletBC(V, f7_upper_func, Bdy_Upper)
bc_f4 = fe.DirichletBC(V, f4_upper_func, Bdy_Upper)
bc_f8 = fe.DirichletBC(V, f8_upper_func, Bdy_Upper)

# Define variational problems

bilinear_forms_stream = []
linear_forms_stream = []

bilinear_forms_collision = []
linear_forms_collision = []

n = fe.FacetNormal(mesh)
opp_idx = {0: 0, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}

alpha = np.arccos( np.sin(theta)*np.sin(theta) )
Omega = 2 * np.sign( np.pi/2 - theta )*np.sqrt(
    np.cos(alpha/3)*( 1 - np.cos(alpha/2) ) )
grad_phi_dot_n = Omega*np.sqrt(2 * kappa * beta)/kappa

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
            - dt*v*fe.Constant(grad_phi_dot_n)*mobility(phi_n)*ds_bottom

lin_form_mu = 4*beta*(phi_n - 1)*(phi_n - 0)*(phi_n - 0.5)*v*fe.dx\
    + kappa*fe.dot(fe.grad(phi_n),fe.grad(v))*fe.dx #- np.sqrt(2*kappa*beta)/kappa\
        #* np.cos(theta)*(phi_n - phi_n**2)*v*fe.ds

for idx in range(Q):

    bilinear_forms_stream.append(f_trial * v * fe.dx)

    double_dot_product_term = -0.5*dt**2 * fe.dot(xi[idx], fe.grad(f_star[idx]))\
        * fe.dot(xi[idx], fe.grad(v)) * fe.dx

    dot_product_force_term = 0.5*dt**2 * fe.dot(xi[idx], fe.grad(v))\
        * body_Force(f_star, phi_n, mu_n, idx) * fe.dx
        

    if idx in opp_idx:
        # UFL scalar: dot product with facet normal
        dot_xi_n = fe.dot(xi[idx], n)

        # indicator = 1.0 when dot_xi_n < 0 (incoming), else 0.0
        indicator = fe.conditional(fe.lt(dot_xi_n, 0.0),
                                   fe.Constant(1.0),
                                   fe.Constant(0.0))

        # build surface term only for incoming distributions
        surface_term = 0.5*dt**2 * v * fe.dot(xi[idx], fe.grad(f_n[opp_idx[idx]])) \
            * dot_xi_n * indicator * fe.ds
    else:
        # no surface contribution for this idx
        surface_term = fe.Constant(0.0) * v * fe.ds

    lin_form_idx = f_star[idx]*v*fe.dx\
        - dt*v*fe.dot(xi[idx], fe.grad(f_star[idx]))*fe.dx\
        + dt*v*body_Force(f_star, phi_n, mu_n, idx)*fe.dx\
        + double_dot_product_term\
        + dot_product_force_term + surface_term

    linear_forms_stream.append(lin_form_idx)

# Assemble matrices for first step

rhs_vec_streaming = [0]*Q
rhs_vec_collision = [0]*Q

sys_mat = []
for idx in range(Q):
    sys_mat.append(fe.assemble(bilinear_forms_stream[idx]))

phi_mat = fe.assemble(bilin_form_AC)
mu_mat = fe.assemble(bilin_form_AC)
rhs_mu = fe.assemble(lin_form_mu)

fe.solve(mu_mat, mu_n.vector(), rhs_mu)

# Timestepping
t = 0.0
for n in range(num_steps):
    t += dt
    
    rhs_AC = fe.assemble(lin_form_AC)
    rhs_mu = fe.assemble(lin_form_mu)
    
    # f_pre_stack = np.array([fi.vector().get_local() for fi in f_n])   # shape (Q,N)
    # rho_pre = np.sum(f_pre_stack, axis=0)
    # momx_pre = np.sum(f_pre_stack * xi_array[:, 0][:, None], axis=0)
    # momy_pre = np.sum(f_pre_stack * xi_array[:, 1][:, None], axis=0)
        
    # f_post_stack = np.zeros_like(f_pre_stack)
    # Perform collision, get post-collision distributions f_i^*
    for idx in range(Q):
        f_eq_vec = f_equil(f_n, idx)
        #f_eq_vec = f_eq.vector().get_local()
        f_n_vec = f_n[idx].vector().get_local()
        
        phi_vec = phi_n.vector().get_local()
        
        tau_vec = phi_vec*tau_h + (1 - phi_vec)*tau_l
        
        f_new = f_n_vec - 1/(tau_vec+0.5) * (f_n_vec - f_eq_vec)
    
        # f_post_stack[idx, :] = f_new
        f_star[idx].vector().set_local(f_new)
        f_star[idx].vector().apply("insert")
        
    # rho_post = np.sum(f_post_stack, axis=0)
    # momx_post = np.sum(f_post_stack * xi_array[:, 0][:, None], axis=0)
    # momy_post = np.sum(f_post_stack * xi_array[:, 1][:, None], axis=0)
    
    # # ---- Compare ----
    # rho_diff = rho_post - rho_pre
    # momx_diff = momx_post - momx_pre
    # momy_diff = momy_post - momy_pre
    # print("max |drho|   =", np.max(np.abs(rho_diff)))
    # print("max |d_momentum_x|=", np.max(np.abs(momx_diff)))
    # print("max |d_momentum_y|=", np.max(np.abs(momy_diff)))

    # Assemble RHS vectors
    for idx in range(Q):
        rhs_vec_streaming[idx] = (fe.assemble(linear_forms_stream[idx]))

    f5_lower_func.vector()[:] = f_n[7].vector()[:]
    f2_lower_func.vector()[:] = f_n[4].vector()[:]
    f6_lower_func.vector()[:] = f_n[8].vector()[:]
    f7_upper_func.vector()[:] = f_n[5].vector()[:]
    f4_upper_func.vector()[:] = f_n[2].vector()[:]
    f8_upper_func.vector()[:] = f_n[6].vector()[:]

    # Apply BCs for distribution functions 5, 2, and 6
    bc_f5.apply(sys_mat[5], rhs_vec_streaming[5])
    bc_f2.apply(sys_mat[2], rhs_vec_streaming[2])
    bc_f6.apply(sys_mat[6], rhs_vec_streaming[6])

    # Apply BCs for distribution functions 7, 4, 8
    bc_f7.apply(sys_mat[7], rhs_vec_streaming[7])
    bc_f4.apply(sys_mat[4], rhs_vec_streaming[4])
    bc_f8.apply(sys_mat[8], rhs_vec_streaming[8])

    # Solve linear system in each timestep, get f^{n+1}
    solver_list = []
    for idx in range(Q):
        A = sys_mat[idx]
        solver = fe.LUSolver(A)
        solver_list.append(solver)
        solver_list[idx].solve(f_nP1[idx].vector(), rhs_vec_streaming[idx])
        
    
    fe.solve(phi_mat, phi_nP1.vector(), rhs_AC)
    fe.solve(mu_mat, mu_n.vector(), rhs_mu)


    # Update previous solutions

    for idx in range(Q):
        f_n[idx].assign(f_nP1[idx])
    phi_n.assign(phi_nP1)
    vel_expr = vel(f_n)
    fe.project(vel_expr, V_vec, function=vel_n)
    
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
        plt.show()
        #plt.close()
        
        a = 1
                


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
