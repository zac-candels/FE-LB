import fenics as fe
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import time
import sys 
import mshr
 
comm = fe.MPI.comm_world
rank = fe.MPI.rank(comm)

start_time = time.time() 

plt.close('all')

# Where to save the plots

fe.parameters["form_compiler"]["optimize"] = True
fe.parameters["form_compiler"]["cpp_optimize"] = True


def computeContactAngle(c_n, h, Cn, mesh):
    
    V = c_n.function_space()
    Vvec = fe.VectorFunctionSpace(mesh, "DG", 0)
    grad_c_fn = fe.project(fe.grad(c_n), Vvec)
    angles = []
    n_vec = np.array([0.0, -1.0])
    
    barycenters = []
    barycenter_vals = []
    for cell in fe.cells(mesh):
        
        midpt = cell.midpoint().array()
        midpt = tuple( (midpt[0], midpt[1]) )
        barycenters.append( midpt )
        barycenter_vals.append( c_n(midpt) )
    
    # Build dictionary
    nodal_dict = {
    tuple(coord): val
    for coord, val in zip(barycenters, barycenter_vals)
    }

    
    # Filter by y-coordinate
    nodal_dict = {
        coord: value
        for coord, value in nodal_dict.items() 
        if coord[1] - surfaceExpr(coord[0]) < 2*h}
    
    # Filter by order parameter value
    nodal_dict = {
        coord: value
        for coord, value in nodal_dict.items() 
        if -0.5 < value < 0.5}
    
    # Determine left-most interfacial point
    min_x = min(coord[0] for coord in nodal_dict.keys())

    # Filter points so we get rid of points near right CL
    nodal_dict = {
        coord: value
        for coord, value in nodal_dict.items() 
        if coord[0] > min_x + 5*Cn}
    
    iter = 0
    for coord, value in nodal_dict.items():
        iter += 1
        #print("coord is", coord)
        grad_c = np.array(grad_c_fn(coord))
        cos_theta = np.dot(grad_c, n_vec) / np.linalg.norm(grad_c)
        angles.append( np.arccos(cos_theta))

    #print("Averaged over ", iter, " points")
        
    theta_avg = np.mean(angles)
    theta_avg = theta_avg * 180 / np.pi
    
    return theta_avg
        
        
        
epsT = 0.05
lamd = 1/6   



T = 20
R0 = 2
initDropDiam = 2*R0
L_x = 6*R0
L_y = 2*R0
nx = 60
ny = 30
h = min(L_x/nx, L_y/ny)

surface_amplitude = 0.5 
surface_freq = L_x/(8*np.pi)

beta_mass_diff = 0.000001


# param_file = sys.argv[1]

# params = {}

# with open(param_file, "r") as f:
#     for line in f:
#         line = line.strip()
#         if line == "" or line.startswith("#"):
#             continue
#         key, value = line.split("=")
#         params[key.strip()] = float(value.strip())

# Re = params["Re"]
# Pe = params["Pe"]
# We = params["We"]
# Cn_param = params["Cn_param"]
# theta_deg = params["theta_deg"]

Pe = 0.1275 
We = 2
Cn_param=  0.05
theta_deg = 30
dt = (1/10)*Cn_param*Pe*h**2
num_steps = int(np.ceil(T/dt))

Cn = initDropDiam * Cn_param

# Lattice speed of sound
c_s = np.sqrt(1/3)
c_s2 = 1/3



theta = theta_deg * np.pi / 180

WORKDIR = os.getcwd()
outDirName = os.path.join(WORKDIR, "CA30") #f"figures_CA{theta_deg}")
os.makedirs(outDirName, exist_ok=True)


tau = 1
xc, yc = L_x/2, R0 - 0.6*R0

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

#mesh = fe.RectangleMesh(comm, fe.Point(0, 0), fe.Point(L_x, L_y), nx, ny, diagonal="crossed")

def surfaceExpr(x):
    
    return surface_amplitude*np.cos(surface_freq*(x - L_x/2) )

domain_n_points = 80
domain_points = []
for n in range(domain_n_points + 1):
    x = n*L_x/domain_n_points
    domain_points.append(fe.Point(x, surfaceExpr(x) ))
domain_points.append(fe.Point(L_x,  surface_amplitude))
domain_points.append(fe.Point(L_x, L_y))
domain_points.append(fe.Point(0., L_y))
domain_points.append(fe.Point(0., surface_amplitude))
domain1 = mshr.Polygon(domain_points)
mesh = mshr.generate_mesh(domain1, 80)

# Set periodic boundary conditions at left and right endpoints


class PeriodicBoundary(fe.SubDomain):
    def inside(self, x, on_boundary):
        return bool(x[0] < fe.DOLFIN_EPS and x[0] > -fe.DOLFIN_EPS and on_boundary)
    def map(self, x, y):
        y[0] = x[0] - L_x
        y[1] = x[1]


pbc = PeriodicBoundary()


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
Vvec = fe.VectorFunctionSpace(mesh, "DG", 0, constrained_domain=pbc)
vel_n = fe.Function(Vvec)
mu_n = fe.Function(V)
rho_fn = fe.Function(V)

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

force_fn = fe.Function(V_vec)


class LowerBoundary(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[1] >= -surface_amplitude and x[1] <= surface_amplitude

class TopBoundary(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and fe.near(x[1], L_y)

class LeftBoundary(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and fe.near(x[0], 0.0)

class RightBoundary(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and fe.near(x[0], L_x)

mesh_function = fe.MeshFunction("size_t", mesh, mesh.topology().dim()-1)

Gamma_1 = LowerBoundary()
Gamma_1.mark(mesh_function, 1)
Gamma_2 = TopBoundary()
Gamma_2.mark(mesh_function, 2)
Gamma_3 = LeftBoundary()
Gamma_3.mark(mesh_function, 3)
Gamma_4 = RightBoundary()
Gamma_4.mark(mesh_function, 4)

ds = fe.Measure('ds', domain=mesh, subdomain_data=mesh_function)
n = fe.FacetNormal(mesh)



# Define dynamic pressure
def getDens(f_list):
    return f_list[0] + f_list[1] + f_list[2] + f_list[3] + f_list[4]\
        + f_list[5] + f_list[6] + f_list[7] + f_list[8]


# Define velocity

def getVel(f_list, force_density):
    distr_fn_sum = f_list[0]*xi[0] + f_list[1]*xi[1] + f_list[2]*xi[2]\
        + f_list[3]*xi[3] + f_list[4]*xi[4] + f_list[5]*xi[5]\
        + f_list[6]*xi[6] + f_list[7]*xi[7] + f_list[8]*xi[8]

    density = getDens(f_list)

    vel_term1 = distr_fn_sum/density

    vel_term2 = force_density * dt / (2 * density)

    return vel_term1 + vel_term2


# Define initial equilibrium distributions
def f_equil_init(vel_idx, force_density):
    rho_init = fe.Constant(1.0)
    rho_expr = fe.Constant(1.0)

    vel_0 = - (dt/2)*force_density/rho_init

    ci = xi[vel_idx]
    ci_dot_u = fe.dot(ci, vel_0)
    return w[vel_idx] * rho_expr * (
        1
        + ci_dot_u / c_s**2
        + ci_dot_u**2 / (2*c_s**4)
        - fe.dot(vel_0, vel_0) / (2*c_s**2)
    )


xi_array = np.array([[float(c.values()[0]), float(c.values()[1])] for c in xi])

def f_equil(f_list, vel_idx, force_density):

    rho = getDens(f_list)
    u   = getVel(f_list, force_density)    
    ci       = xi[vel_idx]
    cu = fe.dot(ci, u)
    u2 = fe.dot(u, u)

    feq = w[vel_idx] * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)

    

    return feq

    

def body_Force(vel, vel_idx, force_density):
    prefactor = w[vel_idx]
    inverse_cs2 = 1 / c_s**2
    inverse_cs4 = 1 / c_s**4

    xi_dot_prod_F = fe.dot( xi[vel_idx], force_density)

    u_dot_prod_F = fe.dot(vel, force_density)

    xi_dot_u = fe.dot(xi[vel_idx], vel)

    Force = prefactor*(inverse_cs2*(xi_dot_prod_F - u_dot_prod_F)
                       + inverse_cs4*xi_dot_u*xi_dot_prod_F)

    return Force


force_density = -(1/We)*phi_n * fe.grad(mu_n)


# # Initialize distribution functions. We will use
# where \bar{u}_0 = u_0 - F\Delta t/( 2 \rho_0 ).
# Here we will take u_0 = 0.

for idx in range(Q):
    f_n[idx] = (fe.project(f_equil_init(idx, force_density), V))
    
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

force_density = -(1/We)*phi_n * fe.grad(mu_n)



# Define boundary conditions.

# For f_5, f_2, and f_6, equilibrium boundary conditions at lower wall
# Since we are applying equilibrium boundary conditions
# and assuming no slip on solid walls, f_i^{eq} reduces to
# \rho * w_i

tol = 1e-8


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

bc_f5 = fe.DirichletBC(V, f5_lower_func, mesh_function, 1)
bc_f2 = fe.DirichletBC(V, f2_lower_func, mesh_function, 1)
bc_f6 = fe.DirichletBC(V, f6_lower_func, mesh_function, 1)

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

bc_f7 = fe.DirichletBC(V, f7_upper_func, mesh_function, 2)
bc_f4 = fe.DirichletBC(V, f4_upper_func, mesh_function, 2)
bc_f8 = fe.DirichletBC(V, f8_upper_func, mesh_function, 2)

# Define variational problems

bilinear_forms_stream = []
linear_forms_stream = []

bilinear_forms_collision = []
linear_forms_collision = []

n = fe.FacetNormal(mesh)
opp_idx = {0: 0, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}

ds_bottom = fe.Measure("ds", domain=mesh, subdomain_data=mesh_function, subdomain_id=1)

bilin_form_AC = f_trial * v * fe.dx
bilin_form_mu = f_trial * v * fe.dx

lin_form_AC = phi_n * v * fe.dx - dt*v*fe.dot(getVel(f_n, force_density), fe.grad(phi_n))*fe.dx\
    - dt*(1/Pe)*v*mu_n*fe.dx - (beta_mass_diff/dt)*mass_diff*fe.sqrt( fe.dot(fe.grad(phi_n), fe.grad(phi_n)) )*v*fe.dx\
        - 0.5*dt**2 * fe.dot(getVel(f_n, force_density), fe.grad(v)) * fe.dot(getVel(f_n, force_density), fe.grad(phi_n)) *fe.dx

lin_form_mu =  (1/Cn)*( phi_n*(phi_n**2 - 1)*v*fe.dx\
    + Cn**2*fe.dot(fe.grad(phi_n),fe.grad(v))*fe.dx\
       - (Cn/(np.sqrt(2)) )*np.cos(theta)*(1 - phi_n**2)*v*ds_bottom  )

for idx in range(Q):

    bilinear_forms_collision.append(f_trial * v * fe.dx)
    bilinear_forms_stream.append(f_trial * v * fe.dx)

    double_dot_product_term = -0.5*dt**2 * fe.dot(xi[idx], fe.grad(f_star[idx]))\
        * fe.dot(xi[idx], fe.grad(v)) * fe.dx

    dot_product_force_term = 0.5*dt**2 * fe.dot(xi[idx], fe.grad(v))\
        * body_Force(getVel(f_n, force_density), idx, force_density) * fe.dx
        

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
        + dt*v*body_Force(getVel(f_star, force_density), idx, force_density)*fe.dx\
        + double_dot_product_term\
        + dot_product_force_term + surface_term
        
    lin_form_coll = (f_n[idx] - dt/tau * (f_n[idx] - f_equil(f_n, idx, force_density)) )*v*fe.dx

    linear_forms_stream.append(lin_form_idx)
    linear_forms_collision.append(lin_form_coll)

# Assemble matrices for first step

rhs_vec_streaming = [fe.assemble(linear_forms_stream[i])
    for i in range(Q)]

rhs_vec_collision = [ fe.assemble(linear_forms_collision[i])
    for i in range(Q)]

sys_mat = []
sys_mat2 = []
for idx in range(Q):
    sys_mat.append(fe.assemble(bilinear_forms_stream[idx]))
    sys_mat2.append(fe.assemble(bilinear_forms_collision[idx]))
    
solver_list = []
solver_list2 = []
for idx in range(Q):
    A = sys_mat[idx]
    A2 = sys_mat2[idx]

    # Create CG solver
    solver = fe.KrylovSolver("cg", "hypre_amg")  # use ILU preconditioner
    solver.set_operator(A)
    
    solver2 = fe.KrylovSolver("cg", "hypre_amg")  # use ILU preconditioner
    solver2.set_operator(A2)

    # Optional: set solver parameters
    prm = solver.parameters
    prm["absolute_tolerance"] = 1e-12
    prm["relative_tolerance"] = 1e-8
    prm["maximum_iterations"] = 1000
    prm["nonzero_initial_guess"] = False

    solver_list.append(solver)
    
    prm2 = solver2.parameters
    prm2["absolute_tolerance"] = 1e-12
    prm2["relative_tolerance"] = 1e-8
    prm2["maximum_iterations"] = 1000
    prm2["nonzero_initial_guess"] = False

    solver_list2.append(solver2)

phi_mat = fe.assemble(bilin_form_AC)
mu_mat = fe.assemble(bilin_form_mu)
phi_solver = fe.KrylovSolver("cg", "hypre_amg")
phi_solver.set_operator(phi_mat)

mu_solver = fe.KrylovSolver("cg", "hypre_amg")
mu_solver.set_operator(mu_mat)

rhs_AC = fe.assemble(lin_form_AC)
rhs_mu = fe.assemble(lin_form_mu)

if rank == 0:
    log_file = open(outDirName + "/simulation_log.txt", "w")
    log_file.write(f"{'% mass change':>15}"
                   f"{'max ||u||':>15}"
                   f"{'theta':>15}"
                   f"{'smallest f':>15}"
                   f"{'smallest f x':>15}"
                   f"{'smallest f y':>15}"
                   f"{'LB mass':>15}\n")
    log_file.flush()


phi_file = fe.XDMFFile(comm, f"{outDirName}/phi.xdmf")
phi_file.parameters["flush_output"] = True
phi_file.parameters["functions_share_mesh"] = True
phi_file.parameters["rewrite_function_mesh"] = False

# mu_file = fe.XDMFFile(comm, f"{outDirName}/mu.xdmf")
# mu_file.parameters["flush_output"] = True
# mu_file.parameters["functions_share_mesh"] = True
# mu_file.parameters["rewrite_function_mesh"] = False

vel_file = fe.XDMFFile(comm, f"{outDirName}/vel.xdmf")
vel_file.parameters["flush_output"] = True
vel_file.parameters["functions_share_mesh"] = True
vel_file.parameters["rewrite_function_mesh"] = False

# Timestepping
t = 0.0
mass_init = fe.assemble(phi_n*fe.dx)
for n in range(num_steps):
    t += dt
    
    #print("n = ", n)
    
    fe.assemble(lin_form_AC, tensor=rhs_AC)
    fe.assemble(lin_form_mu, tensor=rhs_mu)


    f_pre_stack = np.array([fi.vector().get_local() for fi in f_n])   # shape (Q,N)
    rho_pre = np.sum(f_pre_stack, axis=0)
    momx_pre = np.sum(f_pre_stack * xi_array[:, 0][:, None], axis=0)
    momy_pre = np.sum(f_pre_stack * xi_array[:, 1][:, None], axis=0)
        
    f_post_stack = np.zeros_like(f_pre_stack)

    # Perform collision, get post-collision distributions f_i^*
    
    for idx in range(Q):
        fe.assemble(linear_forms_collision[idx], tensor=rhs_vec_collision[idx])
        
    for idx in range(Q):
        solver_list2[idx].solve(f_star[idx].vector(), rhs_vec_collision[idx])


    f_post_stack = np.array([fi.vector().get_local() for fi in f_star])
    rho_post = np.sum(f_post_stack, axis=0)
    momx_post = np.sum(f_post_stack * xi_array[:, 0][:, None], axis=0)
    momy_post = np.sum(f_post_stack * xi_array[:, 1][:, None], axis=0)
    
    # # ---- Compare ----
    rho_diff = rho_post - rho_pre
    momx_diff = momx_post - momx_pre
    momy_diff = momy_post - momy_pre    

    # Assemble RHS vectors
    for idx in range(Q):
        fe.assemble(linear_forms_stream[idx], tensor=rhs_vec_streaming[idx])

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
    

    distr_dict = {}
    #if rank == 0:
    #if fe.MPI.rank(comm) == 0 and os.environ.get("SLURM_PROCID") == "0":
    if 1 == 1:
        if n % 5== 0:  # plot every 10 steps
            
            vel_expr = getVel(f_n, force_density)
            fe.project(vel_expr, Vvec, function=vel_n)
            iteration_time = time.time()
            print("time elapsed ", iteration_time - start_time)
            phi_file.write(phi_n, t)
            vel_file.write(vel_n, t)
            # mu_file.write(mu_n, t)
            
            total_mass = fe.assemble(phi_n*fe.dx)
            percent_mass_change = 100*float(mass_diff)/mass_init
            
            # Determine spatial dimension
            dim = vel_n.geometric_dimension()
            vel_vec = vel_n.vector().get_local()
            # Reshape to (num_nodes, dim)
            vel_vec = vel_vec.reshape((-1, dim))

            # Compute nodal norms
            vel_norm = np.linalg.norm(vel_vec, axis=1)

            # Maximum nodal value
            max_vel = vel_norm.max()
            
            for idx in range(Q):
                f_vec = f_n[idx].vector().get_local()
                min_index = np.argmin(f_vec)
                min_value = f_vec[min_index]
                
                dof_coords = V.tabulate_dof_coordinates().reshape((-1, V.mesh().geometry().dim()))
                min_coord = tuple(dof_coords[min_index])
                
                distr_dict[min_coord] = min_value
                
            min_coord = min(distr_dict, key=distr_dict.get)
            min_distr = distr_dict[min_coord]
            
            rho_expr = sum(fk for fk in f_n)
            fe.project(rho_expr, V, function=rho_fn)

            LB_mass = fe.assemble(rho_fn*fe.dx)
            
            theta_avg = 1#computeContactAngle(phi_n, h, Cn, mesh)
                
            print("theta = ", theta_avg, "\n\n", flush=True)

            log_file.write(f"{percent_mass_change:15.3f}"
                           f"{max_vel:15.6e}"
                           f"{theta_avg:15.2f}"
                           f"{min_distr:15.3f}"
                           f"{min_coord[0]:15.2f}"
                           f"{min_coord[1]:15.2f}"
                           f"{LB_mass:15.3f} \n")
            log_file.flush()

if rank == 0:
    log_file.close()
                