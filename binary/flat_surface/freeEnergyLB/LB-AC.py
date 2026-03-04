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
R0 = 2
initDropDiam = 2*R0
L_x = 2.5*initDropDiam
L_y = 0.6*initDropDiam
nx = 80
ny = 60
h = min(L_x/nx, L_y/ny)
dt = h*CFL / 100
num_steps = int(np.ceil(T/dt))

beta_mass_diff = 0.00000001
Pe = 0.1275
Re = 0.1
Cn = 0.05
We = 1

# Lattice speed of sound
c_s = np.sqrt(1/3)
c_s2 = 1/3


rho_h = 1
rho_l = 1

# Relaxation times for heavier and lighter phases
tau_h = 1
tau_l = 1

theta_deg = 30
theta = theta_deg * np.pi / 180

WORKDIR = os.getcwd()
outDirName = os.path.join(WORKDIR, "LBAC_CA30_remove_vel_projection") #f"figures_CA{theta_deg}")
os.makedirs(outDirName, exist_ok=True)



xc, yc = L_x/2, initDropDiam/2 - 2

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


V = fe.FunctionSpace(mesh, "Lagrange", 1, constrained_domain=pbc)


# Define trial and test functions, as well as
# finite element functions at previous timesteps

f_trial = fe.TrialFunction(V)
phi_trial = fe.TrialFunction(V)
mu_trial = fe.TrialFunction(V)
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
mu_nP1 = fe.Function(V)

# Define FE functions to hold post-collision distributions
f_star = []
for idx in range(Q):
    f_star.append(fe.Function(V))


# Define density
def getDens(phi):
    return rho_h*phi + rho_l*(1 - phi)

def getTau(phi):
    inv_tau = phi/tau_h + (1 - phi)/tau_l
    return 1.0 / inv_tau

# Define dynamic pressure
def getPres(f_list):
    return f_list[0] + f_list[1] + f_list[2] + f_list[3] + f_list[4]\
        + f_list[5] + f_list[6] + f_list[7] + f_list[8]

# Define velocity

def getVel(f_list, phi):
    distr_fn_sum = f_list[0]*xi[0] + f_list[1]*xi[1] + f_list[2]*xi[2]\
        + f_list[3]*xi[3] + f_list[4]*xi[4] + f_list[5]*xi[5]\
        + f_list[6]*xi[6] + f_list[7]*xi[7] + f_list[8]*xi[8]

    density = getDens(phi)
    velocity = distr_fn_sum/(density*c_s**2)

    return velocity


# Define initial equilibrium distributions
def f_equil_init(vel_idx):
    
    # We'll take \bar{p} := 1.0
    return w[vel_idx] 


xi_array = np.array([[float(c.values()[0]), float(c.values()[1])] for c in xi])

def f_equil(f_list, phi, idx):
    """
    Compute equilibrium distribution for direction idx
    Returns a NumPy array (values at all DoFs).
    """
    # Number of DoFs
    N = f_list[0].vector().size()

    # Stack all f_i values: shape (Q, N)
    f_stack = np.array([f.vector().get_local() for f in f_list])

    # Compute pressure at each DoF
    pres = np.sum(f_stack, axis=0)  # shape (N,)
    
    # Compute density at each DoF
    density_ufl = getDens(phi)
    density_fn = fe.project(density_ufl, V)
    density_vec = density_fn.vector().get_local()

    # Compute velocity at each DoF
    ux_vec = np.sum(f_stack * xi_array[:,0][:,None], axis=0) / c_s**2
    uy_vec = np.sum(f_stack * xi_array[:,1][:,None], axis=0) / c_s**2

    u2 = ux_vec**2 + uy_vec**2

    # Compute ci . u for this direction
    cu = xi_array[idx,0]*ux_vec + xi_array[idx,1]*uy_vec
    
    f_eq = w[idx]*( 
        pres + density_vec*c_s2 * ( cu / c_s2 + (cu**2 - c_s2*u2)/(2*c_s2**2) ) )

    return f_eq  # NumPy array


# Define \Gamma
def Gamma_vel(f_list, phi, vel_idx):
    vel_arg = getVel(f_list, phi)
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
    
    fluid_vel = getVel(f_list, phi)
    density = getDens(phi)
    
    velocity_prefactor = xi[vel_idx] - fluid_vel
    
    term1 = fe.grad( density* c_s2 )*( Gamma_vel(f_list, phi, vel_idx)\
                                     - Gamma0(vel_idx) )
    
    term2 = mu * fe.grad(phi) * Gamma_vel(f_list, phi, vel_idx)
    

    return fe.dot( velocity_prefactor, term1 + term2 )




# # Initialize distribution functions. We will use
# where \bar{u}_0 = u_0 - F\Delta t/( 2 \rho_0 ).
# Here we will take u_0 = 0.

for idx in range(Q):
    f_n[idx] = (fe.project(f_equil_init(idx), V))
    
# Initialize \phi
c_init_expr = fe.Expression(
    "-tanh( (sqrt(pow(x[0]-xc,2) + pow(x[1]-yc,2)) - R) / (sqrt(2)*eps) )",
    degree=2,  # polynomial degree used for interpolation
    xc=xc,
    yc=yc,
    R=initDropDiam/2,
    eps=Cn
)

phi_n = fe.interpolate(c_init_expr, V)
mass_diff = fe.Constant(0.0)



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

lin_form_AC = phi_n * v * fe.dx - dt*v*fe.dot(getVel(f_n, phi_n), fe.grad(phi_n))*fe.dx\
    - dt*(1/Pe)*v*mu_n*fe.dx - (beta_mass_diff/dt)*mass_diff*fe.sqrt( fe.dot(fe.grad(phi_n), fe.grad(phi_n)) )*v*fe.dx\
        - 0.5*dt**2 * fe.dot(getVel(f_n, phi_n), fe.grad(v)) * fe.dot(getVel(f_n, phi_n), fe.grad(phi_n)) *fe.dx

lin_form_mu =  (1/Cn)*( phi_n*(phi_n**2 - 1)*v*fe.dx\
    + Cn**2*fe.dot(fe.grad(phi_n),fe.grad(v))*fe.dx\
       - (Cn/(np.sqrt(2)) )*np.cos(theta)*(1 - phi_n**2)*v*ds_bottom  )

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


# outfile = fe.XDMFFile(comm, f"{outDirName}/solution.xdmf")
# outfile.parameters["flush_output"] = True
# outfile.parameters["functions_share_mesh"] = True
# outfile.parameters["rewrite_function_mesh"] = False

# Timestepping
t = 0.0
mass_init = fe.assemble(phi_n*fe.dx)
for n in range(num_steps):
    t += dt
    
    #print("n = ", n)
    
    rhs_AC = fe.assemble(lin_form_AC)
    rhs_mu = fe.assemble(lin_form_mu)

    
    # f_pre_stack = np.array([fi.vector().get_local() for fi in f_n])   # shape (Q,N)
    # rho_pre = np.sum(f_pre_stack, axis=0)
    # momx_pre = np.sum(f_pre_stack * xi_array[:, 0][:, None], axis=0)
    # momy_pre = np.sum(f_pre_stack * xi_array[:, 1][:, None], axis=0)
        
    # f_post_stack = np.zeros_like(f_pre_stack)
    # Perform collision, get post-collision distributions f_i^*
    
    tau_fn = getTau(phi_n)
    tau_func = fe.project(tau_fn, V)
    tau_vec = tau_func.vector().get_local()
    for idx in range(Q):
        f_eq_vec = f_equil(f_n, phi_n, idx)
        #f_eq_vec = f_eq.vector().get_local()
        f_n_vec = f_n[idx].vector().get_local()
        
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

    f5_lower_func.vector()[:] = f_star[7].vector()[:]
    f2_lower_func.vector()[:] = f_star[4].vector()[:]
    f6_lower_func.vector()[:] = f_star[8].vector()[:]
    f7_upper_func.vector()[:] = f_star[5].vector()[:]
    f4_upper_func.vector()[:] = f_star[2].vector()[:]
    f8_upper_func.vector()[:] = f_star[6].vector()[:]

    # # Apply BCs for distribution functions 5, 2, and 6
    bc_f5.apply(sys_mat[5], rhs_vec_streaming[5])
    bc_f2.apply(sys_mat[2], rhs_vec_streaming[2])
    bc_f6.apply(sys_mat[6], rhs_vec_streaming[6])

    # # Apply BCs for distribution functions 7, 4, 8
    bc_f7.apply(sys_mat[7], rhs_vec_streaming[7])
    bc_f4.apply(sys_mat[4], rhs_vec_streaming[4])
    bc_f8.apply(sys_mat[8], rhs_vec_streaming[8])

    # # Solve linear system in each timestep, get f^{n+1}
    for idx in range(Q):
        solver_list[idx].solve(f_nP1[idx].vector(), rhs_vec_streaming[idx])
        
    phi_solver.solve(phi_nP1.vector(), rhs_AC)
    mu_solver.solve(mu_nP1.vector(), rhs_mu)
    


    # Update previous solutions

    for idx in range(Q):
        f_n[idx].assign(f_nP1[idx])
    
    phi_n.assign(phi_nP1)
    mu_n.assign(mu_nP1)
    mass_n = fe.assemble(phi_n*fe.dx)
    mass_diff.assign( (mass_n - mass_init) )
    
    #if fe.MPI.rank(comm) == 0 and os.environ.get("SLURM_PROCID") == "0":
    if 1 == 1:
        if n % 10 == 0:  # plot every 10 steps

            print("n = ", n)
            total_mass = fe.assemble(phi_n*fe.dx)
            print("total mass = ", total_mass, flush=True)
            #outfile.write(phi_n, t)
            print("percent change in mass is ", 100*float(mass_diff)/mass_init, flush=True)

            vel_expr = getVel(f_n, phi_n)
            fe.project(vel_expr, V_vec, function=vel_n)

            vel_vec = vel_n.vector().get_local()

            # Determine spatial dimension
            dim = vel_n.geometric_dimension()

            # Reshape to (num_nodes, dim)
            vel_vec = vel_vec.reshape((-1, dim))

            # Compute nodal norms
            vel_norm = np.linalg.norm(vel_vec, axis=1)

            # Maximum nodal value
            max_vel = vel_norm.max()

            print("Max||u||:", max_vel, flush=True)
            
            f_stack = np.array([f.vector().get_local() for f in f_n])
            
            print("Time elapsed = ", time.time() - start_time, flush=True)

            print("Smallest f val: ", np.min((f_stack)), flush=True )
            print("Smallest f val (in mag)", np.min(np.abs(f_stack)), "\n\n", flush=True)

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
            plt.gca().set_aspect('equal', adjustable='box')
            plt.tight_layout()
            
            # Save the figure to your output folder
            out_file = os.path.join(outDirName, f"phi_t{n:05d}.png")
            plt.savefig(out_file, dpi=200)
            #plt.show()
            plt.close()
                