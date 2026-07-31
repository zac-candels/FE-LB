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
L_x = 8*R0
L_y = 2*R0
nx = 80
ny = 20

A_param = 0.125
kappa = 0.005
interfaceThickness = np.sqrt(kappa/A_param)
M_tilde = 10
theta_deg = 30

# Lattice speed of sound
c_s = np.sqrt(1/3)
c_s2 = 1/3


rho_h = 1
rho_l = 1

# Relaxation times for heavier and lighter phases
tau_h = 1
tau_l = 0.55

theta_deg = 30
theta = theta_deg * np.pi / 180

WORKDIR = os.getcwd()
outDirName = os.path.join(WORKDIR, f"CA{theta_deg}_tauH{tau_h}_tauL{tau_l}_coarseMesh")
if os.path.exists(outDirName):
    shutil.rmtree(outDirName)
os.makedirs(outDirName, exist_ok=True)



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


capTubeLeft      = L_x/4;
capTubeRight     = 3*L_x/4;
capTubeElevation = L_y/2;
capTubeHeight    = L_y/5;

gmsh.initialize()
occ = gmsh.model.occ

# Main rectangle
main = occ.addRectangle(0, 0, 0, L_x, L_y)

# Lower notch
lower = occ.addRectangle(
    capTubeLeft,
    0,
    0,
    capTubeRight-capTubeLeft,
    capTubeElevation-capTubeHeight
)

# Upper notch
upper = occ.addRectangle(
    capTubeLeft,
    capTubeElevation+capTubeHeight,
    0,
    capTubeRight-capTubeLeft,
    L_y-(capTubeElevation+capTubeHeight)
)

occ.cut([(2, main)], [(2, lower), (2, upper)])
occ.synchronize()

gmsh.option.setNumber("Mesh.CharacteristicLengthMin", L_x/200)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", L_x/100)

gmsh.model.mesh.generate(2)
gmsh.write("capillary.msh")

gmsh.finalize()

import meshio

msh = meshio.read("capillary.msh")

triangle_cells = msh.get_cells_type("triangle")

triangle_data = meshio.Mesh(
    points=msh.points[:, :2],
    cells=[("triangle", triangle_cells)]
)

meshio.write("tube.xdmf", triangle_data)

mesh = fe.Mesh()

with fe.XDMFFile("tube.xdmf") as infile:
    infile.read(mesh)

boundary_markers = fe.MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)

h = mesh.hmin()
dt = 0.05*h**2
#dt = 0.0001
beta_mass_diff =  0.1*dt
num_steps = int(np.ceil(T/dt))

class PeriodicBoundary(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and fe.near(x[0], 0.0)

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
    ux_vec[wall_dofs] = 0.0
    uy_vec[wall_dofs] = 0.0

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
        if x[0] <= capTubeLeft - L_x/5:
            values[0] = -1
        elif x[0] > capTubeLeft - L_x/5 and x[0] < capTubeLeft + L_x/15:
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


def fullDomLower(x, on_boundary):
    if on_boundary:
        if fe.near(x[1], 0):
            return True
        else:
            return False
    else:
        return False


rho_expr = sum(fk for fk in f_n)

f5_fullDomLower = f_n[7]  # rho_expr
f2_fullDomLower = f_n[4]  # rho_expr
f6_fullDomLower = f_n[8]  # rho_expr

f5_fullDomLower_func = fe.Function(V)
f2_fullDomLower_func = fe.Function(V)
f6_fullDomLower_func = fe.Function(V)

fe.project(f5_fullDomLower, V, function=f5_fullDomLower_func)
fe.project(f2_fullDomLower, V, function=f2_fullDomLower_func)
fe.project(f6_fullDomLower, V, function=f6_fullDomLower_func)

bc_f5fullDom = fe.DirichletBC(V, f5_fullDomLower_func, fullDomLower)
bc_f2fullDom = fe.DirichletBC(V, f2_fullDomLower_func, fullDomLower)
bc_f6fullDom = fe.DirichletBC(V, f6_fullDomLower_func, fullDomLower)

# Similarly, we will define boundary conditions for f_7, f_4, and f_8
# at the upper wall. Once again, boundary conditions simply reduce
# to \rho * w_i


tol = 1e-8


def fullDomUpper(x, on_boundary):
    if on_boundary:
        if fe.near(x[1], L_y):
            return True
        else:
            return False
    else:
        return False


rho_expr = sum(fk for fk in f_n)

f7_fullDomUpper = f_n[5]  # rho_expr
f4_fullDomUpper = f_n[2]  # rho_expr
f8_fullDomUpper = f_n[6]  # rho_expr

f7_fullDomUpper_func = fe.Function(V)
f4_fullDomUpper_func = fe.Function(V)
f8_fullDomUpper_func = fe.Function(V)

fe.project(f7_fullDomUpper, V, function=f7_fullDomUpper_func)
fe.project(f4_fullDomUpper, V, function=f4_fullDomUpper_func)
fe.project(f8_fullDomUpper, V, function=f8_fullDomUpper_func)

bc_f7fullDom = fe.DirichletBC(V, f7_fullDomUpper_func, fullDomUpper)
bc_f4fullDom = fe.DirichletBC(V, f4_fullDomUpper_func, fullDomUpper)
bc_f8fullDom = fe.DirichletBC(V, f8_fullDomUpper_func, fullDomUpper)

def capTubeLower(x, on_boundary):
    if on_boundary:
        if abs(x[1] - (capTubeElevation-capTubeHeight) ) < 1e-4:
            return True
        else:
            return False
    else:
        return False


rho_expr = sum(fk for fk in f_n)

f5_capTubeLower = f_n[7]  # rho_expr
f2_capTubeLower = f_n[4]  # rho_expr
f6_capTubeLower = f_n[8]  # rho_expr

f5_capTubeLower_func = fe.Function(V)
f2_capTubeLower_func = fe.Function(V)
f6_capTubeLower_func = fe.Function(V)

fe.project(f5_capTubeLower, V, function=f5_capTubeLower_func)
fe.project(f2_capTubeLower, V, function=f2_capTubeLower_func)
fe.project(f6_capTubeLower, V, function=f6_capTubeLower_func)

bc_f5capTube = fe.DirichletBC(V, f5_capTubeLower_func, capTubeLower)
bc_f2capTube = fe.DirichletBC(V, f2_capTubeLower_func, capTubeLower)
bc_f6capTube = fe.DirichletBC(V, f6_capTubeLower_func, capTubeLower)

def capTubeUpper(x, on_boundary):
    if on_boundary:
        if abs(x[1] - (capTubeElevation + capTubeHeight) ) < 1e-4:
            return True
        else:
            return False
    else:
        return False


rho_expr = sum(fk for fk in f_n)

f7_capTubeUpper = f_n[5]  # rho_expr
f4_capTubeUpper = f_n[2]  # rho_expr
f8_capTubeUpper = f_n[6]  # rho_expr

f7_capTubeUpper_func = fe.Function(V)
f4_capTubeUpper_func = fe.Function(V)
f8_capTubeUpper_func = fe.Function(V)

fe.project(f7_capTubeUpper, V, function=f7_capTubeUpper_func)
fe.project(f4_capTubeUpper, V, function=f4_capTubeUpper_func)
fe.project(f8_capTubeUpper, V, function=f8_capTubeUpper_func)

bc_f7capTube = fe.DirichletBC(V, f7_capTubeUpper_func, capTubeUpper)
bc_f4capTube = fe.DirichletBC(V, f4_capTubeUpper_func, capTubeUpper)
bc_f8capTube = fe.DirichletBC(V, f8_capTubeUpper_func, capTubeUpper)

def Bdy_Left(x, on_boundary):
    if on_boundary:
        if fe.near(x[0], capTubeLeft):
            return True
        else:
            return False
    else:
        return False    
    
    
f6_left = f_n[8]  # rho_expr
f3_left = f_n[1]  # rho_expr
f7_left = f_n[5]  # rho_expr

f6_left_func = fe.Function(V)
f3_left_func = fe.Function(V)
f7_left_func = fe.Function(V)

fe.project(f6_left, V, function=f6_left_func)
fe.project(f3_left, V, function=f3_left_func)
fe.project(f7_left, V, function=f7_left_func)

bc_f6Left = fe.DirichletBC(V, f6_left_func, Bdy_Left)
bc_f3Left = fe.DirichletBC(V, f3_left_func, Bdy_Left)
bc_f7Left = fe.DirichletBC(V, f7_left_func, Bdy_Left)

def Bdy_Right(x, on_boundary):
    if on_boundary:
        if fe.near(x[0], capTubeRight):
            return True
        else:
            return False
    else:
        return False    
    
    
f8_right = f_n[6]  # rho_expr
f1_right = f_n[3]  # rho_expr
f5_right = f_n[7]  # rho_expr

f8_right_func = fe.Function(V)
f1_right_func = fe.Function(V)
f5_right_func = fe.Function(V)

fe.project(f8_right, V, function=f8_right_func)
fe.project(f1_right, V, function=f1_right_func)
fe.project(f5_right, V, function=f5_right_func)

bc_f8Right = fe.DirichletBC(V, f8_right_func, Bdy_Right)
bc_f1Right = fe.DirichletBC(V, f1_right_func, Bdy_Right)
bc_f5Right = fe.DirichletBC(V, f5_right_func, Bdy_Right)

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
        return on_boundary and fe.near(x[1], capTubeElevation-capTubeHeight) and x[0] > capTubeLeft and x[0] < capTubeRight
    
# Subdomain for bottom wall
class Top(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and fe.near(x[1], capTubeElevation+capTubeHeight) and x[0] > capTubeLeft and x[0] < capTubeRight

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
    
    tau_fn = getTau(phi_n)
    tau_func = fe.project(tau_fn, V)
    tau_vec = tau_func.vector().get_local()
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

    f5_fullDomLower_func.assign(f_star[7])
    f2_fullDomLower_func.assign( f_star[4])
    f6_fullDomLower_func.assign(f_star[8])
    f5_capTubeLower_func.assign(f_star[7])
    f2_capTubeLower_func.assign( f_star[4])
    f6_capTubeLower_func.assign(f_star[8])
    
    f7_fullDomUpper_func.assign(f_star[5])
    f4_fullDomUpper_func.assign(f_star[2])
    f8_fullDomUpper_func.assign(f_star[6])
    f7_capTubeUpper_func.assign(f_star[5])
    f4_capTubeUpper_func.assign(f_star[2])
    f8_capTubeUpper_func.assign(f_star[6])
    
    
    f6_left_func.assign(f_star[8])
    f3_left_func.assign(f_star[1])
    f7_left_func.assign(f_star[5])
    f5_right_func.assign(f_star[7])
    f1_right_func.assign(f_star[3])
    f8_right_func.assign(f_star[6])
    
    
    # Apply BCs for lower boundary
    bc_f5fullDom.apply(sys_mat[5], rhs_vec_streaming[5])
    bc_f2fullDom.apply(sys_mat[2], rhs_vec_streaming[2])
    bc_f6fullDom.apply(sys_mat[6], rhs_vec_streaming[6])
    
    bc_f5capTube.apply(sys_mat[5], rhs_vec_streaming[5] )
    bc_f2capTube.apply(sys_mat[2], rhs_vec_streaming[2])
    bc_f6capTube.apply(sys_mat[6], rhs_vec_streaming[6])
    
    # Apply BCs for top boundary
    bc_f7fullDom.apply(sys_mat[7], rhs_vec_streaming[7] )
    bc_f4fullDom.apply(sys_mat[4], rhs_vec_streaming[4] )
    bc_f8fullDom.apply(sys_mat[8], rhs_vec_streaming[8] )
    bc_f7capTube.apply(sys_mat[7], rhs_vec_streaming[7] )
    bc_f4capTube.apply(sys_mat[4], rhs_vec_streaming[4] )
    bc_f8capTube.apply(sys_mat[8], rhs_vec_streaming[8] )
    
    bc_f6Left.apply(sys_mat[6], rhs_vec_streaming[6] )
    bc_f3Left.apply(sys_mat[3], rhs_vec_streaming[3] )
    bc_f7Left.apply(sys_mat[7], rhs_vec_streaming[7] )
    
    bc_f8Right.apply(sys_mat[8], rhs_vec_streaming[8] )
    bc_f1Right.apply(sys_mat[1], rhs_vec_streaming[1])
    bc_f5Right.apply(sys_mat[5], rhs_vec_streaming[5])

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
        if n % 1000 == 0:  # plot every 10 steps
            phi_file.write(phi_n, t)
            vel_file.write(vel_n, t)
            print("n = ", n)
            print("total mass = ", mass_n, flush=True)
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

            meniscusPosition = trackMeniscus(phi_n, mesh)


            
            print("x_{meniscus} = ", meniscusPosition)
                
            print("About to write to file", flush = True)
            if rank == 0:
                log_file.write(f"{n:15.3f}"
                                f"{max_vel:15.6e}"
                                f"{meniscusPosition:15.2f}\n")
                log_file.flush()
            print("Wrote to simulation log", flush=True)