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
from scipy      import optimize
import src.postProcessing.computeContactAngle as cca 
 
comm = fe.MPI.comm_world
rank = fe.MPI.rank(comm)

start_time = time.time() 

plt.close('all')

# Where to save the plots

fe.parameters["form_compiler"]["optimize"] = True
fe.parameters["form_compiler"]["cpp_optimize"] = True
      

T = 300
R0 = 2
initDropDiam = 2*R0
L_x = 6*R0
L_y = 6*R0
L_z = 2*R0
nx = 60
ny = 60
nz = 20
h = min(L_x/nx, L_y/ ny)

A = 0.5
kappa = 0.04
interfaceThickness = 0.8#np.sqrt(kappa/A)
tau = 0.6
M_tilde = 10
theta_deg = 30


# Lattice speed of sound
c_s = np.sqrt(1/3)
c_s2 = 1/3



theta = theta_deg * np.pi / 180

WORKDIR = os.getcwd()
outDirName = os.path.join(WORKDIR, f"forceInCollision")
if os.path.exists(outDirName):
    shutil.rmtree(outDirName)
os.makedirs(outDirName, exist_ok=True)


xc, yc, zc = L_x/2, L_y/2, R0 - 0.6*R0


Q = 19
# D3Q19 lattice velocities
xi = [
    fe.Constant(( 0.0,  0.0,  0.0)),  # 0

    # Face-centered directions
    fe.Constant(( 1.0,  0.0,  0.0)),  # 1
    fe.Constant((-1.0,  0.0,  0.0)),  # 2
    fe.Constant(( 0.0,  1.0,  0.0)),  # 3
    fe.Constant(( 0.0, -1.0,  0.0)),  # 4
    fe.Constant(( 0.0,  0.0,  1.0)),  # 5
    fe.Constant(( 0.0,  0.0, -1.0)),  # 6

    # Edge-centered directions
    fe.Constant(( 1.0,  1.0,  0.0)),  # 7
    fe.Constant((-1.0,  1.0,  0.0)),  # 8
    fe.Constant((-1.0, -1.0,  0.0)),  # 9
    fe.Constant(( 1.0, -1.0,  0.0)),  # 10

    fe.Constant(( 1.0,  0.0,  1.0)),  # 11
    fe.Constant((-1.0,  0.0,  1.0)),  # 12
    fe.Constant((-1.0,  0.0, -1.0)),  # 13
    fe.Constant(( 1.0,  0.0, -1.0)),  # 14

    fe.Constant(( 0.0,  1.0,  1.0)),  # 15
    fe.Constant(( 0.0, -1.0,  1.0)),  # 16
    fe.Constant(( 0.0, -1.0, -1.0)),  # 17
    fe.Constant(( 0.0,  1.0, -1.0)),  # 18
]

# Corresponding D3Q19 weights
w = np.array([
    1/3,              # Rest particle
    1/18, 1/18, 1/18, 1/18, 1/18, 1/18,          # Face-centered
    1/36, 1/36, 1/36, 1/36,
    1/36, 1/36, 1/36, 1/36,
    1/36, 1/36, 1/36, 1/36                       # Edge-centered
])

# Set up domain. For simplicity, do unit square mesh.

mesh = fe.BoxMesh(
    comm,
    fe.Point(0.0, 0.0, 0.0),
    fe.Point(L_x, L_y, L_z),
    nx, ny, nz
)

h = mesh.hmin()
dt = 0.1*h**3
#dt = 0.0001
beta_mass_diff =  0.1*dt
num_steps = int(np.ceil(T/dt))
# Set periodic boundary conditions at left and right endpoints


class PeriodicBoundary(fe.SubDomain):
    def inside(self, x, on_boundary):
        return bool(
            on_boundary and
            (
                (x[0] < fe.DOLFIN_EPS) or   # x = 0
                (x[1] < fe.DOLFIN_EPS)      # y = 0
            )
        )

    def map(self, x, y):
        # Initialize all coordinates
        y[0] = x[0]
        y[1] = x[1]
        y[2] = x[2]

        # Periodic in x
        if x[0] > L_x - fe.DOLFIN_EPS:
            y[0] = x[0] - L_x

        # Periodic in y
        if x[1] > L_y - fe.DOLFIN_EPS:
            y[1] = x[1] - L_y


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
forceDensity_z = fe.Function(V)

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
    return sum(f_list)


# Define velocity

def getVel(f_list, force_density):
    momentumDensity = sum(f_list[i] * xi[i] for i in range(len(xi)))

    density = getDens(f_list)

    vel_term1 = momentumDensity/density

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
c_init_expr = fe.Expression(
    "-tanh( (sqrt(pow(x[0]-xc,2) + pow(x[1]-yc,2) + pow(x[2]-zc,2)) - R) / (sqrt(2)*eps) )",
    degree=2,  # polynomial degree used for interpolation
    xc=xc,
    yc=yc,
    zc=zc,
    R=initDropDiam/2,
    eps=interfaceThickness
)

phi_n = fe.interpolate(c_init_expr, V)
mass_diff = fe.Constant(0.0)

force_density = -phi_n * fe.grad(mu_n)

forceDensity_n = fe.project(force_density, V_dis)

# Define boundary conditions.

def Bdy_Lower(x, on_boundary):
    if on_boundary:
        if fe.near(x[2], 0, tol):
            return True
        else:
            return False
    else:
        return False


f5_lower = f_n[6]  
f11_lower = f_n[13]  
f12_lower = f_n[14]
f15_lower = f_n[17]
f16_lower = f_n[18]

f5_lower_func = fe.Function(V)
f11_lower_func = fe.Function(V)
f12_lower_func = fe.Function(V)
f15_lower_func = fe.Function(V)
f16_lower_func = fe.Function(V)

fe.project(f5_lower, V, function=f5_lower_func)
fe.project(f11_lower, V, function=f11_lower_func)
fe.project(f12_lower, V, function=f12_lower_func)
fe.project(f15_lower, V, function=f15_lower_func)
fe.project(f16_lower, V, function=f16_lower_func)

bc_f5 = fe.DirichletBC(V, f5_lower_func, Bdy_Lower)
bc_f11 = fe.DirichletBC(V, f11_lower_func, Bdy_Lower)
bc_f12 = fe.DirichletBC(V, f12_lower_func, Bdy_Lower)
bc_f15 = fe.DirichletBC(V, f15_lower_func, Bdy_Lower)
bc_f16 = fe.DirichletBC(V, f16_lower_func, Bdy_Lower)

# Similarly, we will define boundary conditions for f_7, f_4, and f_8
# at the upper wall. Once again, boundary conditions simply reduce
# to \rho * w_i


tol = 1e-8


def Bdy_Upper(x, on_boundary):
    if on_boundary:
        if fe.near(x[2], L_z, tol):
            return True
        else:
            return False
    else:
        return False


rho_expr = sum(fk for fk in f_n)

f6_upper = f_n[5]  # rho_expr
f13_upper = f_n[11]  # rho_expr
f14_upper = f_n[12]  # rho_expr
f17_upper = f_n[15]
f18_upper = f_n[16]

f6_upper_func = fe.Function(V)
f13_upper_func = fe.Function(V)
f14_upper_func = fe.Function(V)
f17_upper_func = fe.Function(V)
f18_upper_func = fe.Function(V)

fe.project(f6_upper, V, function=f6_upper_func)
fe.project(f13_upper, V, function=f13_upper_func)
fe.project(f14_upper, V, function=f14_upper_func)
fe.project(f17_upper, V, function=f17_upper_func)
fe.project(f18_upper, V, function=f18_upper_func)

bc_f6 = fe.DirichletBC(V, f6_upper_func, Bdy_Upper)
bc_f13 = fe.DirichletBC(V, f13_upper_func, Bdy_Upper)
bc_f14 = fe.DirichletBC(V, f14_upper_func, Bdy_Upper)
bc_f17 = fe.DirichletBC(V, f17_upper_func, Bdy_Upper)
bc_f18 = fe.DirichletBC(V, f18_upper_func, Bdy_Upper)



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
        return on_boundary and fe.near(x[2], 0.0)

bottom = Bottom()
bottom.mark(boundaries, 1)   # assign ID = 1 to bottom boundary
ds_bottom = fe.Measure("ds", domain=mesh, subdomain_data=boundaries, subdomain_id=1)

bilin_form_AC = f_trial * v * fe.dx
bilin_form_mu = f_trial * v * fe.dx


lin_form_AC = - dt*v*fe.dot(getVel(f_n, force_density), fe.grad(phi_n))*fe.dx\
    - dt*M_tilde*v*mu_n*fe.dx - (beta_mass_diff/dt)*mass_diff*fe.sqrt( fe.dot(fe.grad(phi_n), fe.grad(phi_n)) )*v*fe.dx\
        - 0.5*dt**2 * fe.dot(getVel(f_n, force_density), fe.grad(v)) * fe.dot(getVel(f_n, force_density), fe.grad(phi_n)) *fe.dx

lin_form_mu =  A*phi_n*(phi_n**2 - 1)*v*fe.dx\
    + kappa*fe.dot(fe.grad(phi_n),fe.grad(v))*fe.dx\
       + kappa/(np.sqrt(2)*interfaceThickness)*np.cos(theta)*(phi_n**2-1)*v*ds_bottom
       
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
forceVec_z = rhs_mu.copy()

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

xi_arr = np.array([
    [ 0,  0,  0],  # rest

    [ 1,  0,  0],  # axis directions
    [-1,  0,  0],
    [ 0,  1,  0],
    [ 0, -1,  0],
    [ 0,  0,  1],
    [ 0,  0, -1],

    [ 1,  1,  0],  # edge directions
    [-1,  1,  0],
    [-1, -1,  0],
    [ 1, -1,  0],

    [ 1,  0,  1],
    [-1,  0,  1],
    [-1,  0, -1],
    [ 1,  0, -1],

    [ 0,  1,  1],
    [ 0, -1,  1],
    [ 0, -1, -1],
    [ 0,  1, -1]
], dtype=float)

# dof_coords = V.tabulate_dof_coordinates().reshape((-1, 2))
# wall_dofs = np.where(
#     (np.abs(dof_coords[:, 1]) < 1e-10) |
#     (np.abs(dof_coords[:, 1] - L_y) < 1e-10)
# )[0]
# Timestepping
t = 0.0
forceVals_x = []
forceVals_y = []
forceVals_z = []
mass_init = fe.assemble( (phi_n+1)/2*fe.dx)
for n in range(num_steps):
    t += dt
    
    #print("n = ", n)
    prevTimeAcVec.zero()
    fe.as_backend_type(prevTimeAcVec).vec().pointwiseMult(phi_n.vector().vec(), sysMatLumped[0])
    fe.assemble(lin_form_AC, tensor=rhsVecTemp)
    rhs_AC = prevTimeAcVec + rhsVecTemp
    fe.assemble(lin_form_mu, tensor=rhs_mu)
    
    
    projectForceTimeStart = time.time()
    fe.assemble(-phi_n * fe.grad(mu_n)[0]*v*fe.dx, tensor=forceVec_x )
    fe.assemble(-phi_n * fe.grad(mu_n)[1]*v*fe.dx, tensor=forceVec_y)
    fe.assemble(-phi_n * fe.grad(mu_n)[2]*v*fe.dx, tensor=forceVec_z)
    
    fe.solve(massMat, forceDensity_x.vector(), forceVec_x)
    # petscForce_x = fe.as_backend_type(forceVec_x)
    # forceDensity_x.vector().vec().pointwiseDivide(petscForce_x.vec(), M_petsc)
    fe.solve(massMat, forceDensity_y.vector(), forceVec_y)
    
    fe.solve(massMat, forceDensity_z.vector(), forceVec_z)
    # petscForce_y = fe.as_backend_type(forceVec_y)
    # forceDensity_y.vector().vec().pointwiseDivide(petscForce_y.vec(), M_petsc)
    projectForceTimeEnd = time.time()
    #print("project force time = ", projectForceTimeEnd - projectForceTimeStart)
    
    pre_coll_time_lb = time.time()
    # We will try to do collision locally, since it is a pure
    # time-dependnet ODE
    
    f_vals = np.array([f_n[idx].vector().get_local() for idx in range(Q)])
    
    forceVals_x = forceDensity_x.vector().get_local()
    #forceVals_x = forceVals_x.reshape((-1, mesh.geometry().dim()))
    
    forceVals_y = forceDensity_y.vector().get_local()
    #forceVals_y = forceVals_y.reshape((-1, mesh.geometry().dim()))
    
    forceVals_z = forceDensity_z.vector().get_local()

    # Compute rho and u as numpy arrays over all DOFs
    rho = f_vals.sum(axis=0)                          # shape (n_dofs,)
    ux  = (xi_arr[:,0,None] * f_vals).sum(axis=0) / rho + forceVals_x*dt/(2*rho)
    uy  = (xi_arr[:,1,None] * f_vals).sum(axis=0) / rho + forceVals_y*dt/(2*rho)
    uz  = (xi_arr[:,2,None] * f_vals).sum(axis=0) / rho + forceVals_z*dt/(2*rho)
    # ux[wall_dofs] = 0.0
    # uy[wall_dofs] = 0.0
    vel = np.stack([ux, uy, uz])
    cu = xi_arr[:,0,None]*ux + xi_arr[:,1,None]*uy + xi_arr[:,2,None]*uz        # (9, n_dofs)
    u2 = ux**2 + uy**2 + uz**2                                  # (n_dofs,)
    feq = w[:,None] * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)
    
    # Now for the foce term
    u_dot_F = ux * forceVals_x + uy * forceVals_y + uz*forceVals_z  # (n_dofs,)
    ck_dot_F = xi_arr @ np.column_stack((forceVals_x,forceVals_y, forceVals_z)).T   # shape (Q, n_dofs)
    
    ck_dot_u = (
    xi_arr[:,0,None]*ux
    + xi_arr[:,1,None]*uy
    + xi_arr[:,2,None]*uz
    )   # (9, n_dofs) -- same as your 'cu'

    force_term = w[:, None] * (
          ck_dot_F / c_s**2
        + (ck_dot_u * ck_dot_F) / c_s**4   # ← ck_dot_u, not u_dot_F
        - u_dot_F[None, :] / c_s**2        # ← minus sign
    )
    

    f_star_np = f_vals - dt/(tau)*(f_vals - feq) + dt*force_term
    [f_star[idx].vector().set_local(f_star_np[idx,:]) for idx in range(Q)]
    # rho = f_star_np.sum(axis=0)
    # ux  = (xi_arr[:,0,None] * f_vals).sum(axis=0) / rho + forceVals_x*dt/(2*rho)
    # uy  = (xi_arr[:,1,None] * f_vals).sum(axis=0) / rho + forceVals_y*dt/(2*rho)
    # vel_star.vector().set_local(np.stack([ux, uy], axis=1).flatten())

    post_coll_time_lb = time.time()
    #print("collision_time =", post_coll_time_lb - pre_coll_time_lb)


    
    stream_FE_start_time = time.time()
    for idx in range(Q):
        M_lumped.mult(f_star[idx].vector(), streamingPrevTimeVecs[idx])
        advectionMats[idx].mult(f_star[idx].vector(), advectionVecs[idx])
        doubleAdvectionMats[idx].mult(f_star[idx].vector(), doubleAdvectionVecs[idx])

        rhsVecStreaming[idx].zero()
        rhsVecStreaming[idx].axpy(1.0, streamingPrevTimeVecs[idx])
        rhsVecStreaming[idx].axpy(-dt, advectionVecs[idx])
        rhsVecStreaming[idx].axpy(0.5*dt**2, doubleAdvectionVecs[idx])
    stream_FE_end_time = time.time()
    #print("stream FE time = ", stream_FE_end_time - stream_FE_start_time)
    
    # f5_noSlope_func.vector()[:] = f_star[7].vector()[:]
    # f2_noSlope_func.vector()[:] = f_star[4].vector()[:]
    # f6_noSlope_func.vector()[:] = f_star[8].vector()[:]

    assignApplyTimeStart = time.time()
    f5_lower_func.assign(f_star[6])
    f11_lower_func.assign(f_star[13])
    f12_lower_func.assign(f_star[14])
    f15_lower_func.assign(f_star[17])
    f16_lower_func.assign(f_star[18])

    f6_upper_func.assign(f_star[5])
    f13_upper_func.assign(f_star[11])
    f14_upper_func.assign(f_star[12])
    f17_upper_func.assign(f_star[15])
    f18_upper_func.assign(f_star[16])

    solveTimeStart = time.time()
    # # Solve linear system in each timestep, get f^{n+1}
    for idx in range(Q):
        #solver_list[idx].solve(f_nP1[idx].vector(), rhsVecStreaming[idx])
        vi = fe.as_backend_type(rhsVecStreaming[idx]).vec()
        f_nP1[idx].vector().vec().pointwiseDivide(vi, sysMatLumped[idx])
        
    # Apply BCs for lower boundary
    bc_f5.apply( f_nP1[5].vector())
    bc_f11.apply(f_nP1[11].vector())
    bc_f12.apply( f_nP1[12].vector())
    bc_f15.apply(f_nP1[15].vector())
    bc_f16.apply(f_nP1[16].vector())
    
    
    # Apply BCs for top boundary
    bc_f6.apply( f_nP1[6].vector())
    bc_f13.apply( f_nP1[13].vector())
    bc_f14.apply( f_nP1[14].vector())
    bc_f17.apply(f_nP1[17].vector())
    bc_f18.apply(f_nP1[18].vector())
        
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
        if n % 1== 0:  # plot every 10 steps
            print("n = ", n)
            
            
            rho_expr = getDens(f_n)
            fe.project(rho_expr, V, function=rho_n)
            
            vel_expr = getVel(f_n, force_density)
            #fe.project(vel_expr, V_dis, function=vel_dis)
            fe.project(vel_expr, V_cont, function=vel_cont)
            #div_u = fe.project(fe.div(vel_cont), V)
            iteration_time = time.time()
            print("time elapsed ", iteration_time - start_time, "\n")
            phi_file.write(phi_n, t)
            vel_file.write(vel_cont, t)
            pres_file.write(rho_n, t)
            #div_file.write(div_u, t)
            #mu_file.write(mu_n, t)
            

            
            #print("mass_n = ", mass_nP1)
            #massDiffNonLinTerm = fe.assemble(fe.sqrt( fe.dot(fe.grad(phi_n), fe.grad(phi_n)) )*v*fe.dx)
            print("phi max = ", np.max(phi_n.vector().get_local()))
            print("phi min = ", np.min(phi_n.vector().get_local()))
            
            #print("gradPhi norm = ", np.linalg.norm(.vector().get_local()))
            percent_mass_change = 100*float(mass_diff)/mass_init
            print("mass change = ", percent_mass_change, "%")
            
            # Determine spatial dimension
            dim = vel_cont.geometric_dimension()
            vel_vec = vel_cont.vector().get_local()
            # Reshape to (num_nodes, dim)
            vel_vec = vel_vec.reshape((-1, dim))

            # Compute nodal norms
            vel_norm = np.linalg.norm(vel_vec, axis=1)
        

            # Maximum nodal value
            max_vel = vel_norm.max()
            print("umax = ", max_vel)
            for idx in range(Q):
                f_vec = f_n[idx].vector().get_local()
                min_index = np.argmin(f_vec)
                min_value = f_vec[min_index]
                
                dof_coords = V.tabulate_dof_coordinates().reshape((-1, V.mesh().geometry().dim()))
                min_coord = tuple(dof_coords[min_index])
                
                distr_dict[min_coord] = min_value
                
            min_coord = min(distr_dict, key=distr_dict.get)
            min_distr = distr_dict[min_coord]
            
            rho_vals = rho_n.vector().get_local()
            print("max density is", np.max(rho_vals))
            print("min density is", np.min(rho_vals))

            LB_mass = fe.assemble(rho_n*fe.dx)
            
            theta_avg = 1#cca.computeContactAngle_gradPhi(phi_n, h, interfaceThickness, mesh)
            theta_geom = 1#cca.computeContactAngle_heightDiam(phi_n, h, interfaceThickness, mesh)
                
            print("theta avg = ", theta_avg, flush=True)
            print("theta geom = ", theta_geom, "\n\n", flush=True)

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
                