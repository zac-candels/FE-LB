import sys
sys.path.insert(0, "/home/zcandels/FE-LB")
import fenics as fe
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import time 
import mshr
import shutil
import random
from scipy      import optimize
import src.postProcessing.computeContactAngle as cca 
 
comm = fe.MPI.comm_world
rank = fe.MPI.rank(comm)

start_time = time.time() 

plt.close('all')

# Where to save the plots

fe.parameters["form_compiler"]["optimize"] = True
fe.parameters["form_compiler"]["cpp_optimize"] = True
      
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
    min_x = min(coord[0] for coord in nodal_dict.keys())
    
    return min_x


T = 300
R0 = 2
initDropDiam = 2*R0
L_x = 25*R0
L_y = 4*R0
nx = 2000
ny = 30
h = min(L_x/nx, L_y/ ny)

A = 0.5
kappa = 0.02
interfaceThickness = np.sqrt(kappa/A)
tau = 1
M_tilde = 10
theta_deg = 60


# Lattice speed of sound
c_s = np.sqrt(1/3)
c_s2 = 1/3



theta = theta_deg * np.pi / 180

WORKDIR = os.getcwd()
outDirName = os.path.join(WORKDIR, f"capIntrusion_Ibeam_tau{tau}_CA{theta_deg}")
if os.path.exists(outDirName):
    shutil.rmtree(outDirName)
os.makedirs(outDirName, exist_ok=True)


xc, yc = L_x/4 + L_x/8, R0 - 0.6*R0

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

capTubeLeft = L_x/4
capTubeRight = 3*L_x/4
capTubeElevation = L_y/2
capTubeHeight = L_y/10

fullDom = mshr.Rectangle(fe.Point(0.0, 0.0), fe.Point(L_x, L_y))
rectLower = mshr.Rectangle(fe.Point(capTubeLeft, 0),\
                           fe.Point(capTubeRight, capTubeElevation-capTubeHeight) )
rectUpper=  mshr.Rectangle(fe.Point(capTubeLeft, L_y),\
                           fe.Point(capTubeRight, capTubeElevation+capTubeHeight) )

# Subtract circle from rectangle
domain = fullDom - rectLower - rectUpper

# Generate mesh
mesh = mshr.generate_mesh(domain, 90)

boundary_markers = fe.MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)

h = mesh.hmin()
dt = 0.01*h**2
#dt = 0.0001
beta_mass_diff =  0.1*dt
num_steps = int(np.ceil(T/dt))
# Set periodic boundary conditions at left and right endpoints


class PeriodicBoundary(fe.SubDomain):
    def inside(self, point, on_boundary):
        return abs(point[0] - 0.0)<h/2 and on_boundary

    def map(self, right_bdy, left_bdy):
        # Map left boundary to the right
        left_bdy[0] = right_bdy[0] - L_x
        left_bdy[1] = right_bdy[1]
            
pbc = PeriodicBoundary()


V = fe.FunctionSpace(mesh, "Lagrange", 1, constrained_domain=pbc)
dofCoords = V.tabulate_dof_coordinates()
dofCoords = dofCoords.reshape((-1, mesh.geometry().dim()))


# Define trial and test functions, as well as
# finite element functions at previous timesteps

f_trial = fe.TrialFunction(V)
phi_trial = fe.TrialFunction(V)
mu_trial = fe.TrialFunction(V)
f_n = []
for idx in range(Q):
    f_n.append(fe.Function(V))
phi_n = fe.Function(V)
V_cont = fe.VectorFunctionSpace(mesh, "P", 1, constrained_domain=pbc)
V_dis = fe.VectorFunctionSpace(mesh, "DG", 0, constrained_domain=pbc)

vel_star = fe.Function(V_dis)
vel_cont = fe.Function(V_cont)
vel_dis = fe.Function(V_dis)
mu_n = fe.Function(V)
rho_n = fe.Function(V)
forceDensity_x = fe.Function(V)
forceDensity_y = fe.Function(V)

v = fe.TestFunction(V)
v_vec = fe.TestFunction(V_cont)
trial_vec = fe.TrialFunction(V_cont)

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

force_fn = fe.Function(V_dis)




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


force_density = -phi_n * fe.grad(mu_n)


# # Initialize distribution functions. We will use
# where \bar{u}_0 = u_0 - F\Delta t/( 2 \rho_0 ).
# Here we will take u_0 = 0.

for idx in range(Q):
    f_n[idx] = (fe.project(f_equil_init(idx, force_density), V))
    

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



# class InitialConditions(fe.UserExpression):
#     def __init__(self, **kwargs):
#         random.seed(2 + fe.MPI.rank(fe.MPI.comm_world))
#         super().__init__(**kwargs)
#     def eval(self, values, x):
#         if x[0] <= L_x/3:
#             values[0] = 1
#         elif x[0] > L_x - L_x/8:
#             values[0] = 1
#         else:
#             values[0] = -1

#     def value_shape(self):
#         return ()

phi_init = InitialConditions(degree=1)
phi_n.interpolate(phi_init)


mass_diff = fe.Constant(0.0)

force_density = -phi_n * fe.grad(mu_n)

forceDensity_n = fe.project(force_density, V_dis)

# Define boundary conditions.


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
        if fe.near(x[1], capTubeElevation-capTubeHeight):
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
        if fe.near(x[1], capTubeElevation + capTubeHeight):
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


lin_form_AC = - dt*v*fe.dot(getVel(f_n, force_density), fe.grad(phi_n))*fe.dx\
    - dt*M_tilde*v*mu_n*fe.dx - (beta_mass_diff/dt)*mass_diff*fe.sqrt( fe.dot(fe.grad(phi_n), fe.grad(phi_n)) )*v*fe.dx\
        - 0.5*dt**2 * fe.dot(getVel(f_n, force_density), fe.grad(v)) * fe.dot(getVel(f_n, force_density), fe.grad(phi_n)) *fe.dx

lin_form_mu =  A*phi_n*(phi_n**2 - 1)*v*fe.dx\
    + kappa*fe.dot(fe.grad(phi_n),fe.grad(v))*fe.dx\
       + kappa/(np.sqrt(2)*interfaceThickness)*np.cos(theta)*(phi_n**2-1)*v*ds_bottom\
           + kappa/(np.sqrt(2)*interfaceThickness)*np.cos(theta)*(phi_n**2-1)*v*ds_top
       
advection_forms = []
double_advection_forms = []

# Define linear and bilinear forms for the collision and streaming steps
for idx in range(Q):

    bilinear_forms_stream.append(f_trial * v * fe.dx)
    
    advection_forms.append( v*fe.dot(xi[idx], fe.grad(f_trial))*fe.dx )
    double_advection_forms.append( fe.dot(xi[idx],fe.grad(v))\
                                  *fe.dot(xi[idx],fe.grad(f_trial))*fe.dx )


# Assemble matrices for first step

massForm = f_trial*v*fe.dx
massMat = fe.assemble(massForm)
mass_action_form = fe.action(massForm, fe.Constant(1))
M_lumped = fe.assemble(massForm)
M_lumped.zero()
M_lumped.set_diagonal(fe.assemble(mass_action_form))
M_vect = fe.assemble(mass_action_form)
M_petsc = fe.as_backend_type(M_vect).vec()

sys_mat = []
sys_mat2 = []
sysMatLumped = []
advectionMats = []
advectionTransposeMats = []
doubleAdvectionMats = []
for idx in range(Q):
    sysMatLumped.append(M_petsc.copy())
    advectionMats.append(fe.assemble(advection_forms[idx]))
    A_T = advectionMats[idx].copy()
    advectionTransposeMats.append(A_T)
    doubleAdvectionMats.append(fe.assemble(double_advection_forms[idx]))
massMat = fe.assemble(f_trial*v*fe.dx)

streamingPrevTimeVecs= [f_star[0].vector().copy() for _ in range(Q)]
advectionVecs = [f_star[0].vector().copy() for _ in range(Q)]
doubleAdvectionVecs =[f_star[0].vector().copy() for _ in range(Q)]
rhsVecStreaming = [f_star[0].vector().copy() for _ in range(Q)]
prevTimeAcVec = f_star[0].vector().copy()
rhsVecTemp = f_star[0].vector().copy()

xi_arr = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],
                   [1,1],[-1,1],[-1,-1],[1,-1]], dtype=float)

phi_mat = fe.assemble(bilin_form_AC)
mu_mat = fe.assemble(bilin_form_mu)
phi_solver = fe.KrylovSolver("cg", "hypre_amg")
phi_solver.set_operator(phi_mat)

mu_solver = fe.KrylovSolver("cg", "hypre_amg")
mu_solver.set_operator(mu_mat)

rhs_AC = fe.assemble(lin_form_AC)
rhs_mu = fe.assemble(lin_form_mu)

forceVec_x = rhs_mu.copy()
forceVec_y = rhs_mu.copy()

if rank == 0:
    log_file = open(outDirName + "/simulation_log.txt", "w")
    log_file.write(f"{'% mass change':>15}"
                   f"{'max ||u||':>15}"
                   f"{'x_{meniscus}':>15}"
                   f"{'smallest f':>15}"
                   f"{'smallest f x':>15}"
                   f"{'smallest f y':>15}"
                   f"{'LB mass':>15}\n")
    log_file.flush()


phi_file = fe.XDMFFile(comm, f"{outDirName}/phi.xdmf")
phi_file.parameters["flush_output"] = True
phi_file.parameters["functions_share_mesh"] = True
phi_file.parameters["rewrite_function_mesh"] = False

mu_file = fe.XDMFFile(comm, f"{outDirName}/mu.xdmf")
mu_file.parameters["flush_output"] = True
mu_file.parameters["functions_share_mesh"] = True
mu_file.parameters["rewrite_function_mesh"] = False

vel_file = fe.XDMFFile(comm, f"{outDirName}/vel.xdmf")
vel_file.parameters["flush_output"] = True
vel_file.parameters["functions_share_mesh"] = True
vel_file.parameters["rewrite_function_mesh"] = False

div_file = fe.XDMFFile(comm, f"{outDirName}/div.xdmf")
div_file.parameters["flush_output"] = True
div_file.parameters["functions_share_mesh"] = True
div_file.parameters["rewrite_function_mesh"] = False

pres_file = fe.XDMFFile(comm, f"{outDirName}/pres.xdmf")
pres_file.parameters["flush_output"] = True 
pres_file.parameters["functions_share_mesh"] = True 
pres_file.parameters["rewrite_function_mesh"] = False

# # Apply BCs for upSlope boundary
# bc_f5_upSlope.apply(sys_mat[5])
# bc_f2_upSlope.apply(sys_mat[2])
# bc_f6_upSlope.apply(sys_mat[6])
# bc_f3_upSlope.apply(sys_mat[3])

# # Apply BCs for downSlope boundary
# bc_f1_downSlope.apply(sys_mat[1])
# bc_f5_downSlope.apply(sys_mat[5])
# bc_f2_downSlope.apply(sys_mat[2])
# bc_f6_downSlope.apply(sys_mat[6])

# # Apply BCs for top boundary
# bc_f7_upper.apply(sys_mat[7])
# bc_f4_upper.apply(sys_mat[4])
# bc_f8_upper.apply(sys_mat[8])

xi_arr = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],
                   [1,1],[-1,1],[-1,-1],[1,-1]], dtype=float)

dof_coords = V.tabulate_dof_coordinates().reshape((-1, 2))
wall_dofs = np.where(
    (np.abs(dof_coords[:, 1]) < 1e-10) |
    (np.abs(dof_coords[:, 1] - L_y) < 1e-10)
)[0]
# Timestepping
t = 0.0
forceVals_x = []
forceVals_y = []
inverse_cs2 = 1 / c_s**2
inverse_cs4 = 1 / c_s**4
mass_init = fe.assemble( (phi_n+1)/2*fe.dx)
for n in range(num_steps):
    t += dt
    
    #print("n = ", n)
    prevTimeAcVec.zero()
    fe.as_backend_type(prevTimeAcVec).vec().pointwiseMult(phi_n.vector().vec(), sysMatLumped[0])
    fe.assemble(lin_form_AC, tensor=rhsVecTemp)
    rhs_AC = prevTimeAcVec + rhsVecTemp
    fe.assemble(lin_form_mu, tensor=rhs_mu)
    
    pre_coll_time_lb = time.time()
    # We will try to do collision locally, since it is a pure
    # time-dependnet ODE
    
    f_vals = np.array([f_n[idx].vector().get_local() for idx in range(Q)])
    
    fe.assemble(-phi_n * fe.grad(mu_n)[0]*v*fe.dx, tensor=forceVec_x )
    fe.assemble(-phi_n * fe.grad(mu_n)[1]*v*fe.dx, tensor=forceVec_y)
    
    fe.solve(massMat, forceDensity_x.vector(), forceVec_x)

    fe.solve(massMat, forceDensity_y.vector(), forceVec_y)
    
    forceVals_x = forceDensity_x.vector().get_local()
    #forceVals_x = forceVals_x.reshape((-1, mesh.geometry().dim()))
    
    forceVals_y = forceDensity_y.vector().get_local()
    #forceVals_y = forceVals_y.reshape((-1, mesh.geometry().dim()))

    # Compute rho and u as numpy arrays over all DOFs
    rho = f_vals.sum(axis=0)                          # shape (n_dofs,)
    ux  = (xi_arr[:,0,None] * f_vals).sum(axis=0) / rho + forceVals_x*dt/(2*rho)
    uy  = (xi_arr[:,1,None] * f_vals).sum(axis=0) / rho + forceVals_y*dt/(2*rho)
    # ux[wall_dofs] = 0.0
    # uy[wall_dofs] = 0.0
    vel = np.stack([ux, uy])
    cu = xi_arr[:,0,None]*ux + xi_arr[:,1,None]*uy        # (9, n_dofs)
    u2 = ux**2 + uy**2                                    # (n_dofs,)
    feq = w[:,None] * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)
    

    f_star_np = f_vals - dt/(tau)*(f_vals - feq)
    [f_star[idx].vector().set_local(f_star_np[idx,:]) for idx in range(Q)]
    rho = f_star_np.sum(axis=0)
    ux  = (xi_arr[:,0,None] * f_star_np).sum(axis=0) / rho + forceVals_x*dt/(2*rho)
    uy  = (xi_arr[:,1,None] * f_star_np).sum(axis=0) / rho + forceVals_y*dt/(2*rho)
    vel_star.vector().set_local(np.stack([ux, uy], axis=1).flatten())

    post_coll_time_lb = time.time()
    #print("collision_time =", post_coll_time_lb - pre_coll_time_lb)

    u_dot_prod_F = fe.dot(vel_star, force_density)
    
    stream_FE_start_time = time.time()
    for idx in range(Q):
        
        if idx == 0:
            Force = -w[idx]*(inverse_cs2* u_dot_prod_F)

        else:
    
            xi_dot_prod_F = fe.dot( xi[idx], force_density)
    
            xi_dot_u = fe.dot(xi[idx], vel_star)
    
            Force = w[idx]*(inverse_cs2*(xi_dot_prod_F - u_dot_prod_F)
                           + inverse_cs4*xi_dot_u*xi_dot_prod_F)
        
        advectionForceTerm = fe.assemble(
            fe.dot(xi[idx], fe.grad(v))* Force * fe.dx)
            
        basicForceTerm = fe.assemble(v*Force*fe.dx)
        
        M_lumped.mult(f_star[idx].vector(), streamingPrevTimeVecs[idx])
        advectionMats[idx].mult(f_star[idx].vector(), advectionVecs[idx])
        doubleAdvectionMats[idx].mult(f_star[idx].vector(), doubleAdvectionVecs[idx])

        rhsVecStreaming[idx].zero()
        rhsVecStreaming[idx].axpy(1.0, streamingPrevTimeVecs[idx])
        rhsVecStreaming[idx].axpy(-dt, advectionVecs[idx])
        rhsVecStreaming[idx].axpy(0.5*dt**2, doubleAdvectionVecs[idx])
        
        rhsVecStreaming[idx].axpy(dt, basicForceTerm)
        rhsVecStreaming[idx].axpy(0.5*dt**2, advectionForceTerm)
    stream_FE_end_time = time.time()
    #print("stream FE time = ", stream_FE_end_time - stream_FE_start_time)
    
    # f5_noSlope_func.vector()[:] = f_star[7].vector()[:]
    # f2_noSlope_func.vector()[:] = f_star[4].vector()[:]
    # f6_noSlope_func.vector()[:] = f_star[8].vector()[:]

    assignApplyTimeStart = time.time()
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

    # # Apply BCs for distribution functions 5, 2, and 6
    # bc_f5.apply( rhsVecStreaming[5])
    # bc_f2.apply( rhsVecStreaming[2])
    # bc_f6.apply( rhsVecStreaming[6])

    # # # Apply BCs for distribution functions 7, 4, 8
    # bc_f7.apply( rhsVecStreaming[7])
    # bc_f4.apply(rhsVecStreaming[4])
    # bc_f8.apply(rhsVecStreaming[8])

    solveTimeStart = time.time()
    # # Solve linear system in each timestep, get f^{n+1}
    for idx in range(Q):
        #solver_list[idx].solve(f_nP1[idx].vector(), rhsVecStreaming[idx])
        vi = fe.as_backend_type(rhsVecStreaming[idx]).vec()
        f_nP1[idx].vector().vec().pointwiseDivide(vi, sysMatLumped[idx])
        
    # Apply BCs for lower boundary
    bc_f5fullDom.apply( f_nP1[5].vector())
    bc_f2fullDom.apply(f_nP1[2].vector())
    bc_f6fullDom.apply( f_nP1[6].vector())
    
    bc_f5capTube.apply( f_nP1[5].vector())
    bc_f2capTube.apply(f_nP1[2].vector())
    bc_f6capTube.apply( f_nP1[6].vector())
    
    # Apply BCs for top boundary
    bc_f7fullDom.apply( f_nP1[7].vector())
    bc_f4fullDom.apply( f_nP1[4].vector())
    bc_f8fullDom.apply( f_nP1[8].vector())
    bc_f7capTube.apply( f_nP1[7].vector())
    bc_f4capTube.apply( f_nP1[4].vector())
    bc_f8capTube.apply( f_nP1[8].vector())
    
    bc_f6Left.apply(f_nP1[6].vector())
    bc_f3Left.apply(f_nP1[3].vector())
    bc_f7Left.apply(f_nP1[7].vector())
    
    bc_f8Right.apply(f_nP1[8].vector())
    bc_f1Right.apply(f_nP1[1].vector())
    bc_f5Right.apply(f_nP1[5].vector())
    
    
        
    #phi_solver.solve(phi_nP1.vector(), rhs_AC)
    rhsPhiVec = fe.as_backend_type(rhs_AC).vec()
    phi_nP1.vector().vec().pointwiseDivide(rhsPhiVec, sysMatLumped[0])
    rhsMuVec = fe.as_backend_type(rhs_mu).vec()
    #mu_solver.solve(mu_nP1.vector(), rhs_mu)
    mu_nP1.vector().vec().pointwiseDivide(rhsMuVec, sysMatLumped[0])

    solveTimeEnd = time.time()
    #print("solve time = ", solveTimeEnd - solveTimeStart)
    # Update previous solutions

    for idx in range(Q):
        f_n[idx].assign(f_nP1[idx])
    
    # mass_n = fe.assemble(phi_n*fe.dx) 
    # mass_nP1 = fe.assemble(phi_nP1*fe.dx)
    # mass_diff.assign( ( mass_nP1 - mass_n ) )
    
    # print("mass_n = ", mass_n)
    # print("mass_nP1 = ", mass_nP1)
    # print("mass_diff = ", mass_diff.values()[0])
    
    phi_n.assign(phi_nP1)
    mu_n.assign(mu_nP1)
    
    mass_n = fe.assemble( (phi_n+1)/2*fe.dx)
    mass_diff.assign( (mass_n - mass_init) )
    

    

    distr_dict = {}
    #if rank == 0:
    #if fe.MPI.rank(comm) == 0 and os.environ.get("SLURM_PROCID") == "0":
    if n < 40000000:
        if n % 1000== 0:  # plot every 10 steps
        
            if rank == 0:
                print("n = ", n)
            
            
            rho_expr = getDens(f_n)
            fe.project(rho_expr, V, function=rho_n)
            
            vel_expr = getVel(f_n, force_density)
            #fe.project(vel_expr, V_dis, function=vel_dis)
            fe.project(vel_expr, V_cont, function=vel_cont)
            #div_u = fe.project(fe.div(vel_cont), V)
            iteration_time = time.time()
            # print("time elapsed ", iteration_time - start_time, "\n")
            phi_file.write(phi_n, t)
            vel_file.write(vel_cont, t)
            pres_file.write(rho_n, t)
            #div_file.write(div_u, t)
            #mu_file.write(mu_n, t)
            

            
            #print("mass_n = ", mass_nP1)
            #massDiffNonLinTerm = fe.assemble(fe.sqrt( fe.dot(fe.grad(phi_n), fe.grad(phi_n)) )*v*fe.dx)
            # print("phi max = ", np.max(phi_n.vector().get_local()))
            # print("phi min = ", np.min(phi_n.vector().get_local()))
            
            #print("gradPhi norm = ", np.linalg.norm(.vector().get_local()))
            percent_mass_change = 100*float(mass_diff)/mass_init
            # print("mass change = ", percent_mass_change, "%")
            
            # Determine spatial dimension
            dim = vel_cont.geometric_dimension()
            vel_vec = vel_cont.vector().get_local()
            # Reshape to (num_nodes, dim)
            vel_vec = vel_vec.reshape((-1, dim))

            # Compute nodal norms
            vel_norm = np.linalg.norm(vel_vec, axis=1)
        

            # Maximum nodal value
            max_vel = vel_norm.max()
            # print("umax = ", max_vel)
            # for idx in range(Q):
            #     f_vec = f_n[idx].vector().get_local()
            #     min_index = np.argmin(f_vec)
            #     min_value = f_vec[min_index]
                
            #     dof_coords = V.tabulate_dof_coordinates().reshape((-1, V.mesh().geometry().dim()))
            #     min_coord = tuple(dof_coords[min_index])
                
            #     distr_dict[min_coord] = min_value
                
            min_coord = (1,1)#min(distr_dict, key=distr_dict.get)
            min_distr = 1#distr_dict[min_coord]
            
            rho_vals = rho_n.vector().get_local()
            # print("max density is", np.max(rho_vals))
            # print("min density is", np.min(rho_vals))

            LB_mass = fe.assemble(rho_n*fe.dx)
            
            theta_avg = 1#cca.computeContactAngle_gradPhi(phi_n, h, interfaceThickness, mesh)
            theta_geom = 1#cca.computeContactAngle_heightDiam(phi_n, h, interfaceThickness, mesh)
            
            meniscusPosition = trackMeniscus(phi_n, mesh)
            
            # print("x_{meniscus} = ", meniscusPosition)
                
            # print("theta avg = ", theta_avg, flush=True)
            # print("theta geom = ", theta_geom, "\n\n", flush=True)

            log_file.write(f"{percent_mass_change:15.3f}"
                            f"{max_vel:15.6e}"
                            f"{meniscusPosition:15.2f}"
                           f"{min_distr:15.3f}"
                           f"{min_coord[0]:15.2f}"
                           f"{min_coord[1]:15.2f}"
                           f"{LB_mass:15.3f} \n")
            log_file.flush()
            

if rank == 0:
    log_file.close()
                