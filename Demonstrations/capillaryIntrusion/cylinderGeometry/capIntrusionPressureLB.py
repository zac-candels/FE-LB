import fenics as fe
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import time
import shutil
import sys
sys.path.insert(0, "/home/zcandels/FE-LB")
import mshr
import random
import gmsh
 
comm = fe.MPI.comm_world
rank = fe.MPI.rank(comm)

start_time = time.time() 

plt.close('all')

# Where to save the plots

def trackMeniscus(phi_n, mesh):
   
    barycenters = []
    barycenter_vals = []
    for cell in fe.cells(mesh):
        
        midpt = cell.midpoint().array()
        midpt = tuple( (midpt[0], midpt[1]) )
        barycenters.append( midpt )
        barycenter_vals.append( phi_n(midpt) )
    
    # Build dictionary
    nodal_dict = {
    tuple(coord): val
    for coord, val in zip(barycenters, barycenter_vals)
    }

    # Filter by order parameter value
    nodal_dict = {
        coord: value
        for coord, value in nodal_dict.items() 
        if -0.5 < value < 0.5}
    
    nodal_dict = {
        coord: value
        for coord, value in nodal_dict.items()
        if coord[0] > L_x/4}
    

    # Determine left-most interfacial point
    if len(nodal_dict) > 0:
        local_min_x = min(coord[0] for coord in nodal_dict.keys())
    else:
        local_min_x = np.inf

    global_min_x = fe.MPI.min(
        fe.MPI.comm_world,
        local_min_x
    )

    return global_min_x
    




T = 300
R0 = 2
initDropDiam = 2*R0
L_x = 15*R0
L_y = 1*R0
nx = 60
ny = 15

A_param = 0.125
kappa = 0.005
interfaceThickness = np.sqrt(kappa/A_param)
M_tilde = 10
theta_deg = 60
surfTen = np.sqrt(A_param*kappa)

# Lattice speed of sound
c_s = np.sqrt(1/3)
c_s2 = 1/3


rho_h = 1
rho_l = 1

# Relaxation times for heavier and lighter phases
tau_h = 1
tau_l = tau_h/2

theta_deg = 60
theta = theta_deg * np.pi / 180

WORKDIR = os.getcwd()
outDirName = os.path.join(WORKDIR, "test")#f"CA{theta_deg}_tauH{tau_h}_tauL{tau_l}_surfTen{surfTen:1g}")
if os.path.exists(outDirName):
    shutil.rmtree(outDirName)
os.makedirs(outDirName, exist_ok=True)



xc, yc = L_x/3, R0 - 0.6*R0

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


h = mesh.hmin()
dt = 0.001*h**2
#dt = 0.0001
beta_mass_diff =  0.1*dt
num_steps = int(np.ceil(T/dt))
# Set periodic boundary conditions at left and right endpoints

periodicBdyXLeft = L_x/5 
periodicBdyXRight = 4*L_x/5
class PeriodicBoundary(fe.SubDomain):

    def inside(self, x, on_boundary):

        left = fe.near(x[0], 0.0)

        bottom_periodic = (
            fe.near(x[1], 0.0)
            and (x[0] < periodicBdyXLeft or x[0] > periodicBdyXRight)
        )

        return bool((left or bottom_periodic)
                    and on_boundary)

    def map(self, x, y):

        # x-periodicity
        if fe.near(x[0], L_x):
            y[0] = x[0] - L_x
            y[1] = x[1]

        # y-periodicity on selected intervals
        elif (fe.near(x[1], L_y)
              and (x[0] < periodicBdyXLeft or x[0] > periodicBdyXRight)):
            y[0] = x[0]
            y[1] = x[1] - L_y

        else:
            y[0] = x[0]
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
vel_n = fe.Function(V_vec)
mu_n = fe.Function(V)
pres_n = fe.Function(V)

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
    return (1+phi)/2 * rho_h + (1 - phi)/2 * rho_l

def getTau(phi):
    inv_tau = (phi+1)/(2*tau_h) + (1 - phi)/(2*tau_l)
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

dof_coords = V.tabulate_dof_coordinates().reshape((-1, 2))
wall_dofs = np.where(
    (np.abs(dof_coords[:, 1]) < 1e-4) |
    (np.abs(dof_coords[:, 1] - L_y) < 1e-4)
)[0]
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
    ux_vec = np.sum(f_stack * xi_array[:,0][:,None], axis=0) / (density_vec*c_s**2)
    uy_vec = np.sum(f_stack * xi_array[:,1][:,None], axis=0) / (density_vec*c_s**2)
    # ux_vec[wall_dofs] = 0.0
    # uy_vec[wall_dofs] = 0.0

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
class InitialConditions(fe.UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + fe.MPI.rank(fe.MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        if x[0] <= xc:
            values[0] = 1
        elif x[0] > L_x - L_x/8:
            values[0] = 1
        else:
            values[0] = -1

    def value_shape(self):
        return ()

phi_init = InitialConditions(degree=1)
phi_n.interpolate(phi_init)

mass_diff = fe.Constant(0.0)



# Define boundary conditions.

# For f_5, f_2, and f_6, equilibrium boundary conditions at lower wall
# Since we are applying equilibrium boundary conditions
# and assuming no slip on solid walls, f_i^{eq} reduces to
# \rho * w_i

tol = 1e-4

def Bdy_Lower(x, on_boundary):
    if on_boundary:
        if fe.near(x[1], 0.0) and x[0] > periodicBdyXLeft and x[0] < periodicBdyXRight:
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
        if fe.near(x[1], L_y) and x[0] > periodicBdyXLeft and x[0] < periodicBdyXRight:
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
        return on_boundary and fe.near(x[1], 0.0) and x[0] > periodicBdyXLeft and x[0] < periodicBdyXRight
    
# Subdomain for bottom wall
class Top(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and fe.near(x[1], L_y) and x[0] > periodicBdyXLeft and x[0] < periodicBdyXRight

bottom = Bottom()
bottom.mark(boundaries, 1)   # assign ID = 1 to bottom boundary
ds_bottom = fe.Measure("ds", domain=mesh, subdomain_data=boundaries, subdomain_id=1)

top = Top()
top.mark(boundaries, 2)   # assign ID = 1 to bottom boundary
ds_top = fe.Measure("ds", domain=mesh, subdomain_data=boundaries, subdomain_id=2)

bilin_form_AC = f_trial * v * fe.dx
bilin_form_mu = f_trial * v * fe.dx

lin_form_AC = phi_n * v * fe.dx - dt*v*fe.dot(getVel(f_n, phi_n), fe.grad(phi_n))*fe.dx\
    - dt*M_tilde*v*mu_n*fe.dx - (beta_mass_diff/dt)*mass_diff*fe.sqrt( fe.dot(fe.grad(phi_n), fe.grad(phi_n)) )*v*fe.dx\
        - 0.5*dt**2 * fe.dot(getVel(f_n, phi_n), fe.grad(v)) * fe.dot(getVel(f_n, phi_n), fe.grad(phi_n)) *fe.dx

lin_form_mu =  A_param*phi_n*(phi_n**2 - 1)*v*fe.dx\
    + kappa*fe.dot(fe.grad(phi_n),fe.grad(v))*fe.dx\
       + kappa/(np.sqrt(2)*interfaceThickness)*np.cos(theta)*(phi_n**2-1)*v*ds_bottom\
           + kappa/(np.sqrt(2)*interfaceThickness)*np.cos(theta)*(phi_n**2-1)*v*ds_top

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
    solver = fe.LUSolver("mumps")  # use ILU preconditioner
    solver.set_operator(A)

    # Optional: set solver parameters
    # prm = solver.parameters
    # prm["absolute_tolerance"] = 1e-12
    # prm["relative_tolerance"] = 1e-8
    # prm["maximum_iterations"] = 1000
    # prm["nonzero_initial_guess"] = False

    solver_list.append(solver)

phi_mat = fe.assemble(bilin_form_AC)
mu_mat = fe.assemble(bilin_form_mu)
phi_solver = fe.LUSolver("mumps")
phi_solver.set_operator(phi_mat)

mu_solver = fe.LUSolver("mumps")
mu_solver.set_operator(mu_mat)

log_file = open(outDirName + "/simulation_log.txt", "w")
if rank == 0:

    log_file.write(f"{'n':>15}"
                   f"{'max ||u||':>15}"
                   f"{'x_{meniscus}':>15}\n")
    log_file.flush()



phi_file = fe.XDMFFile(comm, f"{outDirName}/phi.xdmf")
phi_file.parameters["flush_output"] = True
phi_file.parameters["functions_share_mesh"] = True
phi_file.parameters["rewrite_function_mesh"] = False

pres_file = fe.XDMFFile(comm, f"{outDirName}/pres.xdmf")
pres_file.parameters["flush_output"] = True
pres_file.parameters["functions_share_mesh"] = True
pres_file.parameters["rewrite_function_mesh"] = False

vel_file = fe.XDMFFile(comm, f"{outDirName}/vel.xdmf")
vel_file.parameters["flush_output"] = True
vel_file.parameters["functions_share_mesh"] = True
vel_file.parameters["rewrite_function_mesh"] = False

# Timestepping
t = 0.0
mass_init = fe.assemble( (phi_n+1)/2*fe.dx)
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
    
    # tau_fn = getTau(phi_n)
    # tau_func = fe.project(tau_fn, V)
    # tau_vec = tau_func.vector().get_local()
    
    phi_local = phi_n.vector().get_local()
    tau_vec = 1/( (phi_local+1)/(2*tau_h) + (1 - phi_local)/(2*tau_l) )
    for idx in range(Q):
        f_eq_vec = f_equil(f_n, phi_n, idx)
        #f_eq_vec = f_eq.vector().get_local()
        f_n_vec = f_n[idx].vector().get_local()
        
        f_new = f_n_vec - dt/(tau_vec) * (f_n_vec - f_eq_vec)
    
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

    f5_lower_func.assign(f_star[7])
    f2_lower_func.assign( f_star[4])
    f6_lower_func.assign(f_star[8])
    f7_upper_func.assign(f_star[5])
    f4_upper_func.assign(f_star[2])
    f8_upper_func.assign(f_star[6])
    
    
    # Apply BCs for lower boundary
    # Apply BCs for lower boundary
    bc_f5.apply(sys_mat[5], rhs_vec_streaming[5])
    bc_f2.apply(sys_mat[2], rhs_vec_streaming[2])
    bc_f6.apply(sys_mat[6], rhs_vec_streaming[6])
    
    # Apply BCs for top boundary
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
    mass_n = fe.assemble( (phi_n+1)/2*fe.dx)
    mass_diff.assign( (mass_n - mass_init) )
    
    #if fe.MPI.rank(comm) == 0 and os.environ.get("SLURM_PROCID") == "0":
    if 1 == 1:
        if n % 2000 == 0:  # plot every 10 steps
        

            f_stack = np.array([f.vector().get_local() for f in f_n])

            # Compute pressure at each DoF
            pres_np = np.sum(f_stack, axis=0) 
            pres_n.vector().set_local(pres_np)
            pres_n.vector().apply("insert")
            pres_file.write(pres_n, t)
            
            phi_file.write(phi_n, t)
            vel_file.write(vel_n, t)
            print("n = ", n)
            print("total mass = ", mass_n, flush=True)
            #outfile.write(phi_n, t)
            print("percent change in mass is ", 100*float(mass_diff)/mass_init, flush=True)

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
            
            print("Time elapsed = ", time.time() - start_time, flush=True)

            print("Smallest f val: ", np.min((f_stack)), flush=True )
            print("Smallest f val (in mag)", np.min(np.abs(f_stack)), "\n\n", flush=True)

            meniscusPosition = trackMeniscus(phi_n, mesh)


            
            print("x_{meniscus} = ", meniscusPosition)
                
            print("About to write to file", flush = True)
            if rank == 0:
                log_file.write(f"{n:15.3f}"
                                f"{max_vel:15.6e}"
                                f"{meniscusPosition:15.2f}\n")
                log_file.flush()
            print("Wrote to simulation log", flush=True)