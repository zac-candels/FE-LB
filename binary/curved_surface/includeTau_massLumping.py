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
    surfaceSlopeAngles = []
    surfaceSlopes=[]
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
        if coord[1] - surfaceExpr(coord[0]) < 1.3*h}
    
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
        if coord[0] < min_x + 5*Cn}
    
    iter = 0
    for coord, value in nodal_dict.items():
        iter += 1
        surfaceSlopes.append(surfExprDeriv(coord[0]))
        
    surfaceSlope_avg = np.mean(surfaceSlopes)
    
    n_vec_raw = np.array([1, -1/surfaceSlope_avg])
    n_vec = n_vec_raw / np.linalg.norm(n_vec_raw)
    
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

surface_amplitude = 0.5 
surface_freq = L_x/(8*np.pi)

beta_mass_diff = 0.000001


Pe = 0.1275 
We = 2
Cn_param=  0.05
theta_deg = 30


Cn = initDropDiam * Cn_param

# Lattice speed of sound
c_s = np.sqrt(1/3)
c_s2 = 1/3



theta = theta_deg * np.pi / 180

WORKDIR = os.getcwd()
outDirName = os.path.join(WORKDIR, f"test")
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

def surfExprDeriv(x):
    return - surface_amplitude*surface_freq*np.sin(surface_freq*(x-L_x/2))

domain_n_points = 60
domain_points = []
for n in range(domain_n_points + 1):
    x = n*L_x/domain_n_points
    domain_points.append(fe.Point(x, surfaceExpr(x) ))
domain_points.append(fe.Point(L_x,  surface_amplitude))
domain_points.append(fe.Point(L_x, L_y))
domain_points.append(fe.Point(0., L_y))
domain_points.append(fe.Point(0., surface_amplitude))
domain1 = mshr.Polygon(domain_points)
if rank == 0:
    mesh = mshr.generate_mesh(domain1, domain_n_points)
    fe.File("mesh.xml") << mesh  # save to disk
mesh = fe.Mesh("mesh.xml")  # load on all ranks

h = mesh.hmin()
dt = (1/2)*Cn_param*Pe*h**2
num_steps = int(np.ceil(T/dt))
# Set periodic boundary conditions at left and right endpoints


class PeriodicBoundary(fe.SubDomain):
    def inside(self, x, on_boundary):
        return bool(x[0] < fe.DOLFIN_EPS and x[0] > -fe.DOLFIN_EPS and on_boundary)
    def map(self, x, y):
        y[0] = x[0] - L_x
        y[1] = x[1]


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
V_vec = fe.VectorFunctionSpace(mesh, "P", 1, constrained_domain=pbc)
Vvec = fe.VectorFunctionSpace(mesh, "DG", 0, constrained_domain=pbc)

vel_star = fe.Function(Vvec)
vel_n = fe.Function(Vvec)
mu_n = fe.Function(V)
rho_fn = fe.Function(V)
forceDensity_x = fe.Function(V)
forceDensity_y = fe.Function(V)

v = fe.TestFunction(V)
v_vec = fe.TestFunction(V_vec)
trial_vec = fe.TrialFunction(V_vec)

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


boundary_markers = fe.MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)

tol = 2*mesh.hmin()

for facet in fe.facets(mesh):
    if facet.exterior():
        n = facet.normal()
        
        # Bottom boundary → normal has negative y component
        if n.y() < -0.1:   # threshold, adjust if needed
            mp = facet.midpoint()
            x = mp.x()
            y = mp.y()
    
            if abs(y - surfaceExpr(x)) < tol:
                slope = surfExprDeriv(x)
    
                if slope > 1e-3:
                    print("boundary marker found 1")
                    boundary_markers[facet] = 1
                elif slope < -1e-3:
                    print("boundary marker found 2")
                    boundary_markers[facet] = 2
                else:
                    print("boundary marker found 3")
                    boundary_markers[facet] = 3




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

forceDensity_n = fe.project(force_density, V_vec)

# Define boundary conditions.

# Boundary condition for up-slope portion of wall.
# Here,
# f_5^{n+1} \gets f_7^n
# f_2^{n+1} \gets f_4^n
# f_6^{n+1} \gets f_8^n
# f_3^{n+1} \gets f_1^n 
#

rho_expr = sum(fk for fk in f_n)

f5_upSlope = f_n[7] 
f2_upSlope = f_n[4] 
f6_upSlope = f_n[8] 
f3_upSlope = f_n[1]

f5_upSlope_func = fe.Function(V)
f2_upSlope_func = fe.Function(V)
f6_upSlope_func = fe.Function(V)
f3_upSlope_func = fe.Function(V)

fe.project(f5_upSlope, V, function=f5_upSlope_func)
fe.project(f2_upSlope, V, function=f2_upSlope_func)
fe.project(f6_upSlope, V, function=f6_upSlope_func)

bc_f5_upSlope = fe.DirichletBC(V, f5_upSlope_func, boundary_markers, 1)
bc_f2_upSlope = fe.DirichletBC(V, f2_upSlope_func, boundary_markers, 1)
bc_f6_upSlope = fe.DirichletBC(V, f6_upSlope_func, boundary_markers, 1)
bc_f3_upSlope = fe.DirichletBC(V, f3_upSlope_func, boundary_markers, 1)


# We do a similar procedure for the downslope part of the boundary

# Boundary condition for up-slope portion of wall.
# Here,
# f_1^{n+1} \gets f_3^n
# f_5^{n+1} \gets f_7^n
# f_2^{n+1} \gets f_4^n
# f_6^{n+1} \gets f_8^n 

f1_downSlope = f_n[3]
f5_downSlope = f_n[7] 
f2_downSlope = f_n[4] 
f6_downSlope = f_n[8] 

f1_downSlope_func = fe.Function(V)
f5_downSlope_func = fe.Function(V)
f2_downSlope_func = fe.Function(V)
f6_downSlope_func = fe.Function(V)

fe.project(f1_downSlope, V, function=f1_downSlope_func)
fe.project(f5_downSlope, V, function=f5_downSlope_func)
fe.project(f2_downSlope, V, function=f2_downSlope_func)
fe.project(f6_downSlope, V, function=f6_downSlope_func)

bc_f1_downSlope = fe.DirichletBC(V, f1_downSlope_func, boundary_markers, 2)
bc_f5_downSlope = fe.DirichletBC(V, f5_downSlope_func, boundary_markers, 2)
bc_f2_downSlope = fe.DirichletBC(V, f2_downSlope_func, boundary_markers, 2)
bc_f6_downSlope = fe.DirichletBC(V, f6_downSlope_func, boundary_markers, 2)


# Finally if the slope is sufficiently small

f5_noSlope = f_n[7]  # rho_expr
f2_noSlope = f_n[4]  # rho_expr
f6_noSlope = f_n[8]  # rho_expr

f5_noSlope_func = fe.Function(V)
f2_noSlope_func = fe.Function(V)
f6_noSlope_func = fe.Function(V)

fe.project(f5_noSlope, V, function=f5_noSlope_func)
fe.project(f2_noSlope, V, function=f2_noSlope_func)
fe.project(f6_noSlope, V, function=f6_noSlope_func)

bc_f5_noSlope = fe.DirichletBC(V, f5_noSlope_func, boundary_markers, 3)
bc_f2_noSlope = fe.DirichletBC(V, f2_noSlope_func, boundary_markers, 3)
bc_f6_noSlope = fe.DirichletBC(V, f6_noSlope_func, boundary_markers, 3)

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

bc_f7_upper = fe.DirichletBC(V, f7_upper_func, Bdy_Upper)
bc_f4_upper = fe.DirichletBC(V, f4_upper_func, Bdy_Upper)
bc_f8_upper = fe.DirichletBC(V, f8_upper_func, Bdy_Upper)


# Define variational problems

bilinear_forms_stream = []
linear_forms_stream = []

bilinear_forms_collision = []
linear_forms_collision = []

n = fe.FacetNormal(mesh)
opp_idx = {0: 0, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}

ds = fe.Measure("ds", domain=mesh, subdomain_data=boundary_markers)
ds_bottom = ds(1) + ds(2)

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
        + double_dot_product_term\
        + surface_term
        
    lin_form_coll = (f_n[idx] - dt/tau * (f_n[idx] - f_equil(f_n, idx, force_density)) )*v*fe.dx

    linear_forms_stream.append(lin_form_idx)
    linear_forms_collision.append(lin_form_coll)

# Assemble matrices for first step

rhs_vec_streaming = [fe.assemble(linear_forms_stream[i])
    for i in range(Q)]

rhs_vec_collision = [ fe.assemble(linear_forms_collision[i])
    for i in range(Q)]

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
sysMatLumped = M_petsc
for idx in range(Q):
    sys_mat.append(fe.assemble(bilinear_forms_stream[idx]))
    sys_mat2.append(fe.assemble(bilinear_forms_collision[idx]))
mass_mat = fe.assemble(f_trial*v*fe.dx)

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

forceVec_x = rhs_mu.copy()
forceVec_y = rhs_mu.copy()

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

# Apply BCs for upSlope boundary
bc_f5_upSlope.apply(sys_mat[5])
bc_f2_upSlope.apply(sys_mat[2])
bc_f6_upSlope.apply(sys_mat[6])
bc_f3_upSlope.apply(sys_mat[3])

# Apply BCs for downSlope boundary
bc_f1_downSlope.apply(sys_mat[1])
bc_f5_downSlope.apply(sys_mat[5])
bc_f2_downSlope.apply(sys_mat[2])
bc_f6_downSlope.apply(sys_mat[6])

# Apply BCs for top boundary
bc_f7_upper.apply(sys_mat[7])
bc_f4_upper.apply(sys_mat[4])
bc_f8_upper.apply(sys_mat[8])

xi_arr = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],
                   [1,1],[-1,1],[-1,-1],[1,-1]], dtype=float)

# Timestepping
t = 0.0
forceVals_x = []
forceVals_y = []
mass_init = fe.assemble(phi_n*fe.dx)
for n in range(num_steps):
    t += dt
    
    #print("n = ", n)
    
    fe.assemble(lin_form_AC, tensor=rhs_AC)
    fe.assemble(lin_form_mu, tensor=rhs_mu)
    
    pre_coll_time_lb = time.time()
    
    fe.assemble(-(1/We)*phi_n * fe.grad(mu_n)[0]*v*fe.dx, tensor=forceVec_x )
    fe.assemble(-(1/We)*phi_n * fe.grad(mu_n)[1]*v*fe.dx, tensor=forceVec_y)
    
    fe.solve(mass_mat, forceDensity_x.vector(), forceVec_x)
    fe.solve(mass_mat, forceDensity_y.vector(), forceVec_y)
    

    # We will try to do collision locally, since it is a pure
    # time-dependnet ODE
    
    f_vals = np.array([f_n[idx].vector().get_local() for idx in range(Q)])
    
    forceVals_x = forceDensity_x.vector().get_local()
    #forceVals_x = forceVals_x.reshape((-1, mesh.geometry().dim()))
    
    forceVals_y = forceDensity_y.vector().get_local()
    #forceVals_y = forceVals_y.reshape((-1, mesh.geometry().dim()))

    # Compute rho and u as numpy arrays over all DOFs
    rho = f_vals.sum(axis=0)                          # shape (n_dofs,)
    ux  = (xi_arr[:,0,None] * f_vals).sum(axis=0) / rho + forceVals_x*dt/(2*rho)
    uy  = (xi_arr[:,1,None] * f_vals).sum(axis=0) / rho + forceVals_y*dt/(2*rho)
    vel = np.stack([ux, uy])
    cu = xi_arr[:,0,None]*ux + xi_arr[:,1,None]*uy        # (9, n_dofs)
    u2 = ux**2 + uy**2                                    # (n_dofs,)
    feq = w[:,None] * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)
    
    # Now for the foce term
    u_dot_F = ux * forceVals_x + uy * forceVals_y   # (n_dofs,)
    ck_dot_F = xi_arr @ np.column_stack((forceVals_x,forceVals_y)).T   # shape (Q, n_dofs)
    
    force_term = (
    (1/c_s**2) * ck_dot_F
    + (1/c_s**4) * ck_dot_F * u_dot_F[None, :]
    + (1/c_s**2) * u_dot_F[None, :]
    )

    force_term *= w[:, None]
    

    f_star_np = f_vals - (dt/tau)*(f_vals - feq) + dt*force_term
    [f_star[idx].vector().set_local(f_star_np[idx,:]) for idx in range(Q)]
    vel_star.vector().set_local(np.stack([ux, uy], axis=1).flatten())
    post_coll_time_lb = time.time()
    #print("collision_time =", post_coll_time_lb - pre_coll_time_lb)


    
    stream_FE_start_time = time.time()
    for idx in range(Q):
        fe.assemble(linear_forms_stream[idx], tensor=rhs_vec_streaming[idx])
    stream_FE_end_time = time.time()
    #print("stream FE time = ", stream_FE_end_time - stream_FE_start_time)
    
    # f5_noSlope_func.vector()[:] = f_star[7].vector()[:]
    # f2_noSlope_func.vector()[:] = f_star[4].vector()[:]
    # f6_noSlope_func.vector()[:] = f_star[8].vector()[:]


    # Apply BCs for upSlope boundary
    bc_f5_upSlope.apply( rhs_vec_streaming[5])
    bc_f2_upSlope.apply( rhs_vec_streaming[2])
    bc_f6_upSlope.apply( rhs_vec_streaming[6])
    bc_f3_upSlope.apply( rhs_vec_streaming[3])
    
    # Apply BCs for downSlope boundary
    bc_f1_downSlope.apply( rhs_vec_streaming[1])
    bc_f5_downSlope.apply( rhs_vec_streaming[5])
    bc_f2_downSlope.apply( rhs_vec_streaming[2])
    bc_f6_downSlope.apply( rhs_vec_streaming[6])
    
    # Apply BCs for top boundary
    bc_f7_upper.apply( rhs_vec_streaming[7])
    bc_f4_upper.apply( rhs_vec_streaming[4])
    bc_f8_upper.apply( rhs_vec_streaming[8])

    # Apply BCs for noSlope boundary
    # bc_f5_noSlope.apply(sys_mat[5], rhs_vec_streaming[5])
    # bc_f2_noSlope.apply(sys_mat[2], rhs_vec_streaming[2])
    # bc_f6_noSlope.apply(sys_mat[6], rhs_vec_streaming[6])

    # # Solve linear system in each timestep, get f^{n+1}
    for idx in range(Q):
        #solver_list[idx].solve(f_nP1[idx].vector(), rhs_vec_streaming[idx])
        vi = fe.as_backend_type(rhs_vec_streaming[idx]).vec()
        f_nP1[idx].vector().vec().pointwiseDivide(vi, sysMatLumped)
        
    # Apply BCs for upSlope boundary
    bc_f5_upSlope.apply( f_nP1[5].vector())
    bc_f2_upSlope.apply( f_nP1[2].vector())
    bc_f6_upSlope.apply( f_nP1[6].vector())
    bc_f3_upSlope.apply( f_nP1[3].vector())
    
    # Apply BCs for downSlope boundary
    bc_f1_downSlope.apply( f_nP1[1].vector())
    bc_f5_downSlope.apply( f_nP1[5].vector())
    bc_f2_downSlope.apply( f_nP1[2].vector())
    bc_f6_downSlope.apply( f_nP1[6].vector())
    
    # Apply BCs for top boundary
    bc_f7_upper.apply( f_nP1[7].vector())
    bc_f4_upper.apply( f_nP1[4].vector())
    bc_f8_upper.apply( f_nP1[8].vector())
        
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
        if n % 2000== 0:  # plot every 10 steps
            
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
            
            # for idx in range(Q):
            #     f_vec = f_n[idx].vector().get_local()
            #     min_index = np.argmin(f_vec)
            #     min_value = f_vec[min_index]
                
            #     dof_coords = V.tabulate_dof_coordinates().reshape((-1, V.mesh().geometry().dim()))
            #     min_coord = tuple(dof_coords[min_index])
                
            #     distr_dict[min_coord] = min_value
                
            min_coord = (1,1)#min(distr_dict, key=distr_dict.get)
            min_distr = 1#distr_dict[min_coord]
            
            # rho_expr = sum(fk for fk in f_n)
            # fe.project(rho_expr, V, function=rho_fn)

            LB_mass = 1#fe.assemble(rho_fn*fe.dx)
            
            theta_avg = computeContactAngle(phi_n, h, Cn, mesh)
                
            print("theta = ", theta_avg, "\n\n", flush=True)

            # log_file.write(f"{percent_mass_change:15.3f}"
            #                f"{max_vel:15.6e}"
            #                f"{theta_avg:15.2f}"
            #                f"{min_distr:15.3f}"
            #                f"{min_coord[0]:15.2f}"
            #                f"{min_coord[1]:15.2f}"
            #                f"{LB_mass:15.3f} \n")
            # log_file.flush()

if rank == 0:
    log_file.close()
                