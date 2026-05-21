import fenics as fe
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import shutil

comm = fe.MPI.comm_world
rank = fe.MPI.rank(comm)

plt.close('all')

Tfinal = 300
R0 = 2
initDropDiam = 2*R0
L_x = 8*R0
L_y = 4*R0
nx = 80
ny = 40
h = min(L_x/nx, L_y/ny)

mesh = fe.RectangleMesh(comm, fe.Point(0, 0), fe.Point(L_x, L_y),
                        nx, ny, diagonal="crossed")

h = mesh.hmin()
Pe = 0.1275 
We = 2
Cn_param=  0.05
Re = 1
theta_deg = 30
dt = Cn_param*Pe*h**2
beta_mass_diff = 0.00001
num_steps = int(np.ceil(Tfinal/dt))

Cn = initDropDiam * Cn_param
xc, yc = L_x/2, R0 - 0.6*R0

theta = theta_deg * np.pi / 180

WORKDIR = os.getcwd()
outDirName = os.path.join(WORKDIR, f"test")
matPlotFigs = outDirName + "/matPlotFigs"
if os.path.exists(outDirName):
    shutil.rmtree(outDirName)
os.makedirs(matPlotFigs, exist_ok=True)
os.makedirs(outDirName, exist_ok=True)

# Lattice speed of sound
c_s = np.sqrt(1/3) # np.sqrt( 1./3. * h**2/dt**2 )

#nu = 1.0/6.0
#tau = nu/c_s**2 + dt/2 
tau = 0.6

# Number of discrete velocities
Q = 9


#Force prefactor 
alpha_plus = ( 2/dt + 1/tau )
alpha_minus = ( 2/dt - 1/tau )

# Density on wall
rho_wall = 1.0
# Initial density 
rho_init = 1.0
u_wall = (0.0, 0.0)


# D2Q9 lattice velocities
xi = [
    fe.Constant(( 0.0,  0.0)),
    fe.Constant(( 1.0,  0.0)),
    fe.Constant(( 0.0,  1.0)),
    fe.Constant((-1.0,  0.0)),
    fe.Constant(( 0.0, -1.0)),
    fe.Constant(( 1.0,  1.0)),
    fe.Constant((-1.0,  1.0)),
    fe.Constant((-1.0, -1.0)),
    fe.Constant(( 1.0, -1.0)),
]

# Corresponding weights
w = np.array([
    4/9,
    1/9, 1/9, 1/9, 1/9,
    1/36, 1/36, 1/36, 1/36
])

# Set up domain. For simplicity, do unit square mesh.

mesh = fe.RectangleMesh(fe.Point(0,0), fe.Point(L_x, L_y), nx, nx )

# Set periodic boundary conditions at left and right endpoints
class PeriodicBoundaryX(fe.SubDomain):
    def inside(self, x, on_boundary):
        return fe.near(x[0], 0.0) and on_boundary

    def map(self, x, y):
        # Map left boundary to the right
        y[0] = x[0] - L_x
        y[1] = x[1]

pbc = PeriodicBoundaryX()


V = fe.FunctionSpace(mesh, "P", 1, constrained_domain=pbc)




# Define trial and test functions, as well as 
# finite element functions at previous timesteps

f_trial = []
phi_trial = fe.TrialFunction(V)
mu_trial = fe.TrialFunction(V)
f_n = []
mu_n = fe.Function(V)
phi_n = fe.Function(V)
for idx in range(Q):
    f_trial.append(fe.TrialFunction(V))
    f_n.append(fe.Function(V))
    
v = fe.TestFunction(V)
phi_nP1 = fe.Function(V)
mu_nP1 = fe.Function(V)

Vvec = fe.VectorFunctionSpace(mesh, "DG", 0, constrained_domain=pbc)

vel_star = fe.Function(Vvec)
vel_n = fe.Function(Vvec)
# Define density
def getRho(f_list):
    return f_list[0] + f_list[1] + f_list[2] + f_list[3] + f_list[4]\
        + f_list[5] + f_list[6] + f_list[7] + f_list[8]

# Define velocity
def getVel(f_list, Force_density):
    distr_fn_sum = f_list[0]*xi[0] + f_list[1]*xi[1] + f_list[2]*xi[2]\
        + f_list[3]*xi[3] + f_list[4]*xi[4] + f_list[5]*xi[5]\
            + f_list[6]*xi[6] + f_list[7]*xi[7] + f_list[8]*xi[8]
            
    density = getRho(f_list)
    
    vel_term1 = distr_fn_sum/density
    
    vel_term2 = Force_density * dt / (2 * density)
    
    
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

# Define equilibrium distribution
def f_equil(f_list, vel_idx, Force_density):
    rho_expr = sum(fj for fj in f_list)
    u_expr   = getVel(f_list, Force_density)    
    ci       = xi[vel_idx]
    ci_dot_u = fe.dot(ci, u_expr)
    return w[vel_idx] * rho_expr * (
        1
        + ci_dot_u / c_s**2
        + ci_dot_u**2 / (2*c_s**4)
        - fe.dot(u_expr, u_expr) / (2*c_s**2)
    )

def f_equil_extrap(f_list_n, f_list_n_1, vel_idx, Force_density):
    rho_expr = sum(fj for fj in f_list_n)
    u_expr   = getVel(f_list_n, Force_density)    
    ci       = xi[vel_idx]
    ci_dot_u = fe.dot(ci, u_expr)
    
    f_equil_n = w[vel_idx] * rho_expr * (
        1
        + ci_dot_u / c_s**2
        + ci_dot_u**2 / (2*c_s**4)
        - fe.dot(u_expr, u_expr) / (2*c_s**2)
    )
    
    rho_expr = sum(fj for fj in f_list_n_1)
    u_expr   = getVel(f_list_n_1, Force_density)   
    ci       = xi[vel_idx]
    ci_dot_u = fe.dot(ci, u_expr)
    
    f_equil_n_1 = w[vel_idx] * rho_expr * (
        1
        + ci_dot_u / c_s**2
        + ci_dot_u**2 / (2*c_s**4)
        - fe.dot(u_expr, u_expr) / (2*c_s**2)
    )
    
    return 2 * f_equil_n - f_equil_n_1
    
    

# Define collision operator
def coll_op(f_list, vel_idx):
    return -( f_list[vel_idx] - f_equil(f_list, vel_idx, Force_density) ) / tau

def body_Force(vel, vel_idx, Force_density):
    prefactor = w[vel_idx]
    inverse_cs2 = 1 / c_s**2
    inverse_cs4 = 1 / c_s**4
    
    xi_dot_prod_F = xi[vel_idx][0]*Force_density[0]\
        + xi[vel_idx][1]*Force_density[1]
        
    u_dot_prod_F = vel[0]*Force_density[0] + vel[1]*Force_density[1]
    
    xi_dot_u = xi[vel_idx][0]*vel[0] + xi[vel_idx][1]*vel[1]
    
    Force = prefactor*( inverse_cs2*(xi_dot_prod_F - u_dot_prod_F)\
                       + inverse_cs4*xi_dot_u*xi_dot_prod_F)
        
    return Force

def body_Force_extrap(f_list_n, f_list_n_1, vel_idx, Force_density):
    vel_n = getVel(f_list_n, Force_density)
    vel_n_1 = getVel(f_list_n_1, Force_density)
    
    prefactor = w[vel_idx]
    inverse_cs2 = 1 / c_s**2
    inverse_cs4 = 1 / c_s**4
    
    # Compute F^n
    xi_dot_prod_F_n = xi[vel_idx][0]*Force_density[0]\
        + xi[vel_idx][1]*Force_density[1]
        
    u_dot_prod_F_n = vel_n[0]*Force_density[0] + vel_n[1]*Force_density[1]
    
    xi_dot_u_n = xi[vel_idx][0]*vel_n[0] + xi[vel_idx][1]*vel_n[1]
    
    Force_n = prefactor*( inverse_cs2*(xi_dot_prod_F_n - u_dot_prod_F_n)\
                       + inverse_cs4*xi_dot_u_n*xi_dot_prod_F_n)
        
    # Compute F^{n-1}
    xi_dot_prod_F_n_1 = xi[vel_idx][0]*Force_density[0]\
        + xi[vel_idx][1]*Force_density[1]
        
    u_dot_prod_F_n_1 = vel_n_1[0]*Force_density[0] + vel_n_1[1]*Force_density[1]
    
    xi_dot_u_n_1 = xi[vel_idx][0]*vel_n_1[0] + xi[vel_idx][1]*vel_n_1[1]
    
    
        
    Force_n_1 = prefactor*( inverse_cs2*(xi_dot_prod_F_n_1 - u_dot_prod_F_n_1)\
                       + inverse_cs4*xi_dot_u_n_1*xi_dot_prod_F_n_1)
        
    return 2*Force_n - Force_n_1
    
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
Force_density = -(1/We)*phi_n * fe.grad(mu_n)
# # Initialize distribution functions. We will use 
# f_i^{0} \gets f_i^{0, eq}( \rho_0, \bar{u}_0 ),
# where \bar{u}_0 = u_0 - F\Delta t/( 2 \rho_0 ).
# Here we will take u_0 = 0.

for idx in range(Q):
    f_n[idx] = (  fe.project(f_equil_init(idx, Force_density), V )  )


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
    
rho_expr = sum( fk for fk in f_n )
 
f5_lower = f_n[7] # rho_expr
f2_lower = f_n[4] # rho_expr 
f6_lower = f_n[8] # rho_expr

f5_lower_func = fe.Function(V)
f2_lower_func = fe.Function(V)
f6_lower_func = fe.Function(V)

fe.project( f5_lower, V, function=f5_lower_func )
fe.project( f2_lower, V, function=f2_lower_func )
fe.project( f6_lower, V, function=f6_lower_func )

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

rho_expr = sum( fk for fk in f_n )
 
f7_upper = f_n[5] # rho_expr
f4_upper = f_n[2] # rho_expr 
f8_upper = f_n[6] # rho_expr

f7_upper_func = fe.Function(V)
f4_upper_func = fe.Function(V)
f8_upper_func = fe.Function(V)

fe.project( f7_upper, V, function=f7_upper_func )
fe.project( f4_upper, V, function=f4_upper_func )
fe.project( f8_upper, V, function=f8_upper_func )

bc_f7 = fe.DirichletBC(V, f7_upper_func, Bdy_Upper)
bc_f4 = fe.DirichletBC(V, f4_upper_func, Bdy_Upper)
bc_f8 = fe.DirichletBC(V, f8_upper_func, Bdy_Upper)

# Define variational problems


# Define FE functions to hold solution at nP1 timesteps
f_nP1 = []
for idx in range(Q):
    f_nP1.append(fe.Function(V))

f_nM1 = []
for idx in range(Q):
    f_nM1.append( fe.Function(V) ) 
    

bilinear_forms_step2 = []
linear_forms_step1 = []
linear_forms_step2 = []

bilin_form_AC = phi_trial * v * fe.dx
bilin_form_mu = mu_trial * v * fe.dx

# Create MeshFunction for boundary markers
boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)

# Subdomain for bottom wall
class Bottom(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and fe.near(x[1], 0.0)

bottom = Bottom()
bottom.mark(boundaries, 1)   # assign ID = 1 to bottom boundary
ds_bottom = fe.Measure("ds", domain=mesh, subdomain_data=boundaries, subdomain_id=1)

lin_form_AC = phi_n * v * fe.dx - dt*v*fe.dot(getVel(f_n, Force_density), fe.grad(phi_n))*fe.dx\
    - dt*(1/Pe)*v*mu_n*fe.dx - (beta_mass_diff/dt)*mass_diff*fe.sqrt( fe.dot(fe.grad(phi_n), fe.grad(phi_n)) )*v*fe.dx\
        - 0.5*dt**2 * fe.dot(getVel(f_n, Force_density), fe.grad(v)) * fe.dot(getVel(f_n, Force_density), fe.grad(phi_n)) *fe.dx

lin_form_mu =  (1/Cn)*( phi_n*(phi_n**2 - 1)*v*fe.dx\
    + Cn**2*fe.dot(fe.grad(phi_n),fe.grad(v))*fe.dx\
        + (1/np.sqrt(2)*Cn)*np.cos(theta)*(phi_n**2 -1)*v*ds_bottom  )

# Define variational problems for step 2 (CN timestep)

for idx in range(Q):
    bilinear_forms_step2.append( alpha_plus**2*f_trial[idx]*v*fe.dx\
        + alpha_plus*fe.dot( xi[idx], fe.grad(v) ) * f_trial[idx] * fe.dx\
            + alpha_plus*fe.dot( xi[idx], fe.grad(f_trial[idx]) )*v*fe.dx\
                + fe.dot( xi[idx], fe.grad(f_trial[idx]) )\
                    *fe.dot( xi[idx], fe.grad(v) )*fe.dx )

    body_force_np1 = body_Force_extrap(f_n, f_nM1, idx, Force_density)
    body_force_n = body_Force(getVel(f_n, Force_density), idx, Force_density)
    
    linear_forms_step1.append( ( alpha_minus*alpha_plus*f_n[idx]*v\
        + alpha_minus*f_n[idx]*fe.dot( xi[idx], fe.grad(v) )\
        +   (1/tau)*( f_equil_extrap(f_n, f_n, idx, Force_density) + f_equil(f_n, idx, Force_density) ) * alpha_plus*v\
        + (1/tau)*( f_equil_extrap(f_n, f_n, idx, Force_density) + f_equil(f_n, idx, Force_density) ) * fe.dot( xi[idx], fe.grad(v) )\
            - fe.dot( xi[idx], fe.grad(f_n[idx]) )*alpha_plus*v\
                - fe.dot( xi[idx], fe.grad(f_n[idx]) )*fe.dot( xi[idx], fe.grad(v) )\
                    + 0.5*(body_force_n + body_force_n)*alpha_plus*v\
                        + 0.5*(body_force_n + body_force_n)\
                            *fe.dot( xi[idx], fe.grad(v) ) )*fe.dx )

    linear_forms_step2.append( ( alpha_minus*alpha_plus*f_n[idx]*v\
        + alpha_minus*f_n[idx]*fe.dot( xi[idx], fe.grad(v) )\
        +   (1/tau)*( f_equil_extrap(f_n, f_nM1, idx, Force_density) + f_equil(f_n, idx, Force_density) ) * alpha_plus*v\
        + (1/tau)*( f_equil_extrap(f_n, f_nM1, idx, Force_density) + f_equil(f_n, idx, Force_density) ) * fe.dot( xi[idx], fe.grad(v) )\
            - fe.dot( xi[idx], fe.grad(f_n[idx]) )*alpha_plus*v\
                - fe.dot( xi[idx], fe.grad(f_n[idx]) )*fe.dot( xi[idx], fe.grad(v) )\
                    + 0.5*(body_force_np1 + body_force_n)*alpha_plus*v\
                        + 0.5*(body_force_np1 + body_force_n)\
                            *fe.dot( xi[idx], fe.grad(v) ) )*fe.dx )


# Assemble matrices for CN timestep
sys_mat_step2 = []
rhs_vec_step2 = [0]*Q
for idx in range(Q):
    sys_mat_step2.append( fe.assemble(bilinear_forms_step2[idx] ) )
    

phi_mat = fe.assemble(bilin_form_AC)
mu_mat = fe.assemble(bilin_form_mu)
phi_solver = fe.LUSolver(phi_mat)
mu_solver = fe.LUSolver(mu_mat)

rhs_AC = fe.assemble(lin_form_AC)
rhs_mu = fe.assemble(lin_form_mu)

phi_file = fe.XDMFFile(comm, f"{outDirName}/phi.xdmf")
phi_file.parameters["flush_output"] = True
phi_file.parameters["functions_share_mesh"] = True
phi_file.parameters["rewrite_function_mesh"] = False


vel_file = fe.XDMFFile(comm, f"{outDirName}/vel.xdmf")
vel_file.parameters["flush_output"] = True
vel_file.parameters["functions_share_mesh"] = True
vel_file.parameters["rewrite_function_mesh"] = False
    
# CN timestepping
t = 0
mass_init = fe.assemble( (phi_n + 1)/2 * fe.dx)

for n in range(0, num_steps):
    t += dt
    
    rhs_AC = fe.assemble(lin_form_AC)
    rhs_mu = fe.assemble(lin_form_mu)
    
    phi_solver.solve(phi_nP1.vector(), rhs_AC)
    mu_solver.solve(mu_nP1.vector(), rhs_mu)
    # Assemble RHS vectors
    if n == 0:
        for idx in range(Q):
            rhs_vec_step2[idx] = ( fe.assemble(linear_forms_step1[idx]) )
    else:
        for idx in range(Q):
            rhs_vec_step2[idx] = ( fe.assemble(linear_forms_step2[idx]) )
        
    # Apply BCs for distribution functions 5, 2, and 6
    bc_f5.apply(sys_mat_step2[5], rhs_vec_step2[5])
    bc_f2.apply(sys_mat_step2[2], rhs_vec_step2[2])
    bc_f6.apply(sys_mat_step2[6], rhs_vec_step2[6])
    
    # Apply BCs for distribution functions 7, 4, 8
    bc_f7.apply(sys_mat_step2[7], rhs_vec_step2[7])
    bc_f4.apply(sys_mat_step2[4], rhs_vec_step2[4])
    bc_f8.apply(sys_mat_step2[8], rhs_vec_step2[8])
    
    # Solve linear system in each timestep
    for idx in range(Q):
        fe.solve( sys_mat_step2[idx], f_nP1[idx].vector(), rhs_vec_step2[idx] )
        
    # Update previous solutions
    for idx in range(Q):
        f_nM1[idx].assign( f_n[idx] )
    
    for idx in range(Q):
        f_n[idx].assign( f_nP1[idx] )
        
    phi_n.assign(phi_nP1)
    mu_n.assign(mu_nP1)
    mass_n = fe.assemble( (phi_n+1)/2*fe.dx)
    mass_diff.assign( (mass_n - mass_init) )
    
    f5_lower_func.assign(f_n[7])
    f2_lower_func.assign(f_n[4])
    f6_lower_func.assign(f_n[8])
    f7_upper_func.assign(f_n[5])
    f4_upper_func.assign(f_n[2])
    f8_upper_func.assign(f_n[6])
    # fe.project(f_n[4], V, function=f2_lower_func)
    # fe.project(f_n[8], V, function=f6_lower_func)
    # fe.project(f_n[5], V, function=f7_upper_func)
    # fe.project(f_n[2], V, function=f4_upper_func)
    # fe.project(f_n[6], V, function=f8_upper_func)
    
    if n%10  == 0:
            print("n = ", n)
            vel_expr = getVel(f_n, Force_density)
            fe.project(vel_expr, Vvec, function=vel_n)
            phi_file.write(phi_n, t)
            vel_file.write(vel_n, t)
            #print("phi max = ", np.max(phi_n.vector().get_local()) )

        

    
    

