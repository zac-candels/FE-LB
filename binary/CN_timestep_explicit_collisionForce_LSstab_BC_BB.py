import fenics as fe
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

T = 3000
dt = 1
num_steps = int(np.ceil(T/dt))


Re = 0.96
nx = ny = 5
L_x = 30
L_y = 30
h = L_x/nx

error_vec = []

# Lattice speed of sound
c_s = np.sqrt(1/3) # np.sqrt( 1./3. * h**2/dt**2 )

#nu = 1.0/6.0
#tau = nu/c_s**2 + dt/2 
tau_ns = 1
tau_ch = 0.1

# Number of discrete velocities
Q = 9
Force_density = np.array([2.6041666e-5, 0.0])


#Force prefactor 
alpha_plus_g = ( 2/dt + 1/tau_ns )
alpha_minus_g = ( 2/dt - 1/tau_ns )

# Density on wall
rho_wall = 1.0
# Initial density 
rho_init = 1.0
u_wall = (0.0, 0.0)

# Liquid density
rho_l = 1.0
# Air density
rho_g = 0.001

# Mobility
M = 0.2

radius = 10
drop_center_x = L_x / 2
drop_center_y = 2


nu = tau_ch/3
u_max = Force_density[0]*L_y**2/(8*rho_init*nu)


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

g_trial = []
g_n = []
h_trial = []
h_n = []
for idx in range(Q):
    g_trial.append(fe.TrialFunction(V))
    g_n.append(fe.Function(V))
    
    h_trial.append(fe.TrialFunction(V))
    h_n.append(fe.Function(V))
    
v = fe.TestFunction(V)


# Define dynamic pressure
def dynPres(g_list):
    return g_list[0] + g_list[1] + g_list[2] + g_list[3] + g_list[4]\
        + g_list[5] + g_list[6] + g_list[7] + g_list[8]
        
# Define order parameter
def orderParam(h_list):
    return h_list[0] + h_list[1] + h_list[2] + h_list[3] + h_list[4]\
        + h_list[5] + h_list[6] + h_list[7] + h_list[8]
        
# Define density 
def rho(rho_l, rho_g, orderParam):
    return orderParam*rho_l + (1 - orderParam)*rho_g

# Define velocity
def vel(g_list, rho):
    momentum = g_list[0]*xi[0] + g_list[1]*xi[1] + g_list[2]*xi[2]\
        + g_list[3]*xi[3] + g_list[4]*xi[4] + g_list[5]*xi[5]\
            + g_list[6]*xi[6] + g_list[7]*xi[7] + g_list[8]*xi[8]
            
    vel = momentum / rho
    
    return vel


# Define initial equilibrium distributions
def g_equil_init(vel_idx, Force_density):
    rho_init = fe.Constant(1.0)
    rho_expr = fe.Constant(1.0)

    vel_0 = -fe.Constant( ( Force_density[0]*dt/(2*rho_init),
                           Force_density[1]*dt/(2*rho_init) ) )
    
    # u_expr = fe.project(V_vec, vel_0)
    
    ci = xi[vel_idx]
    ci_dot_u = fe.dot(ci, vel_0)
    return w[vel_idx] * rho_expr * (
        1
        + ci_dot_u / c_s**2
        + ci_dot_u**2 / (2*c_s**4)
        - fe.dot(vel_0, vel_0) / (2*c_s**2)
    )

# Define equilibrium distribution
def g_equil(g_list, vel_idx, rho):
    dynPressure = dynPres(g_list)
    u_expr   = vel(g_list, rho)    
    ci       = xi[vel_idx]
    ci_dot_u = fe.dot(ci, u_expr)
    
    vel_terms = ci_dot_u / c_s**2\
    + ci_dot_u**2 / (2*c_s**4)\
    - fe.dot(u_expr, u_expr) / (2*c_s**2)
    
    g_eq = w[i] * ( dynPressure + rho * c_s**2 * vel_terms )
    
    return g_eq

def g_equil_extrap(g_list_n, g_list_nM1, vel_idx, rho_n, rho_nM1):
    
    g_equil_n = g_equil(g_list_n, vel_idx, rho_n)
    
    g_equil_nM1 = g_equil(g_list_nM1, vel_idx, rho_nM1)
    
    return 2 * g_equil_n - g_equil_nM1
    
    
# Define equilibrium for h distributions 
def h_equil(g_list, h_list, vel_idx, rho):
    orderParameter = orderParam(h_list)
    u_expr   = vel(g_list, rho)    
    ci       = xi[vel_idx]
    ci_dot_u = fe.dot(ci, u_expr)
    
    vel_terms = ci_dot_u / c_s**2\
    + ci_dot_u**2 / (2*c_s**4)\
    - fe.dot(u_expr, u_expr) / (2*c_s**2)
    
    return w[i] * orderParameter * (1 + vel_terms)

def h_equil_extrap(g_list_n, g_list_nM1, h_list_n, h_list_nM1,
                   vel_idx, rho_n, rho_nM1):
    
    h_eq_n = h_equil(g_list_n, h_list_n, vel_idx, rho_n)
    
    h_eq_nM1 = h_equil(g_list_nM1, h_list_nM1, vel_idx, rho_nM1)
    
    return 2 * h_eq_n - h_eq_nM1

def Gamma_fn(h_equil, vel, vel_idx, orderParameter):
    
    if vel == 0 :
        return w[vel_idx]
    else:
        return h_equil / orderParameter
    
    
def rel_time( orderParameter ):
    
    inv_tau = orderParameter / tau_ns + ( 1 - orderParameter) / tau_ch
    
    tau = 1/inv_tau
    
    return tau

# Define collision operator
def coll_op_g(g_list, vel_idx, orderParameter):
    
    tau = rel_time(orderParameter)
    
    return -( g_list[vel_idx] - g_equil(g_list, vel_idx) ) / tau 


def coll_op_h(h_list, vel_idx, orderParameter):
    
    tau = rel_time(orderParameter)
    
    return -( h_list[vel_idx] - h_equil(h_list, vel_idx) ) / tau 


def body_Force_g(vel, vel_idx, rho, mu, orderParameter, h_equil):
    
    Gamma_u = Gamma_fn( h_equil, vel, vel_idx, orderParameter)
    Gamma_0 = Gamma_fn( h_equil, 0, vel_idx, orderParameter)
    
    rho_grad = fe.grad( rho )
    orderParam_grad = fe.grad( orderParameter )
    
    dot_prod_arg1 = xi[vel_idx] - vel 
    
    dot_prod_arg2 = rho_grad * c_s**2 * (Gamma_u - Gamma_0)\
        + mu * orderParam_grad * Gamma_u
    
    Force_g = fe.dot( dot_prod_arg1, dot_prod_arg2)
    
    return Force_g

def body_Force_g_extrap(vel_n, vel_nM1, vel_idx, rho_n, rho_nM1,
                        mu_n, mu_nM1, orderParameter_n, orderParameter_nM1,
                        h_equil_n, h_equil_nM1):
    
    F_g_n = body_Force_g(vel_n, vel_idx, rho_n, mu_n, orderParameter_n, h_equil_n)
    
    F_g_nM1 = body_Force_g(vel_nM1, vel_idx, rho_nM1, mu_nM1,
                           orderParameter_nM1, h_equil_nM1)
    
    return 2 * F_g_n - F_g_nM1
    
    
    
def body_Force_h(vel, vel_idx, rho, mu, orderParameter, h_equil, lam, dyn_pres):
    
    Gamma_u = Gamma_fn( h_equil, vel, vel_idx, orderParameter)
    
    orderParam_grad = fe.grad( orderParameter )
    pressure_grad = fe.grad(dyn_pres)
    
    dot_prod_arg1 = xi[vel_idx] - vel 
    
    dot_prod_arg2 = orderParam_grad - orderParameter / (rho * c_s**2)\
        * ( pressure_grad - mu * orderParam_grad ) * Gamma_u
        
    term1 = fe.dot( dot_prod_arg1, dot_prod_arg2 )
    
    term2 = M * lam * Gamma_u
    
    return term1 + term2

def body_Force_h_extrap(vel_n, vel_nM1, vel_idx, rho_n, rho_nM1, mu_n, mu_nM1,
                        orderParameter_n, orderParameter_nM1,
                        h_equil_n, h_equil_nM1, lam_n, lam_nM1,
                        dyn_pres_n, dyn_pres_nM1):
    
    F_h_n = body_Force_h(vel_n, vel_idx, rho_n, mu_n, orderParameter_n,
                         h_equil_n, lam_n, dyn_pres_n)
    
    F_h_nM1 = body_Force_h(vel_nM1, vel_idx, rho_nM1, mu_nM1,
                           orderParameter_nM1, h_equil_nM1, lam_nM1, dyn_pres_nM1)
    
    return 2 * F_h_n - F_h_nM1
    
    
    
# # Initialize distribution functions. We will use 
# g_i^{0} \gets g_i^{0, eq}( \rho_0, \bar{u}_0 ),
# where \bar{u}_0 = u_0 - F\Delta t/( 2 \rho_0 ).
# Here we will take u_0 = 0.

for idx in range(Q):
    g_n[idx] = (  fe.project(g_equil_init(idx, Force_density), V )  )
    

# Also need to initialize the h_distributions

def init_orderParam(x, y):
    r2 = (x - drop_center_x)**2 + (y - drop_center_y )**2
    
    C_0 = 0.5 - 0.5 * np.tanh( 2 * (np.sqrt(r2) - radius) )/4.0
    
    C_0_fenics = fe.project(C_0, V)
    
    return C_0_fenics

    


# Define boundary conditions.

# For g_5, g_2, and g_6, equilibrium boundary conditions at lower wall
# Since we are applying equilibrium boundary conditions 
# and assuming no slip on solid walls, g_i^{eq} reduces to
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
    
rho_expr = sum( fk for fk in g_n )
 
g5_lower = g_n[7] # rho_expr
g2_lower = g_n[4] # rho_expr 
g6_lower = g_n[8] # rho_expr

g5_lower_func = fe.Function(V)
g2_lower_func = fe.Function(V)
g6_lower_func = fe.Function(V)

fe.project( g5_lower, V, function=g5_lower_func )
fe.project( g2_lower, V, function=g2_lower_func )
fe.project( g6_lower, V, function=g6_lower_func )

bc_g5 = fe.DirichletBC(V, g5_lower_func, Bdy_Lower)
bc_g2 = fe.DirichletBC(V, g2_lower_func, Bdy_Lower)
bc_g6 = fe.DirichletBC(V, g6_lower_func, Bdy_Lower)

# Similarly, we will define boundary conditions for g_7, g_4, and g_8
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

rho_expr = sum( gk for gk in g_n )
 
g7_upper = g_n[5] # rho_expr
g4_upper = g_n[2] # rho_expr 
g8_upper = g_n[6] # rho_expr

g7_upper_func = fe.Function(V)
g4_upper_func = fe.Function(V)
g8_upper_func = fe.Function(V)

fe.project( g7_upper, V, function=g7_upper_func )
fe.project( g4_upper, V, function=g4_upper_func )
fe.project( g8_upper, V, function=g8_upper_func )

bc_g7 = fe.DirichletBC(V, g7_upper_func, Bdy_Upper)
bc_g4 = fe.DirichletBC(V, g4_upper_func, Bdy_Upper)
bc_g8 = fe.DirichletBC(V, g8_upper_func, Bdy_Upper)

# Define variational problems


# Define FE functions to hold solution at nP1 timesteps
g_nP1 = []
h_nP1 = []
mu_nP1 = []
lam_nP1 = []
for idx in range(Q):
    g_nP1.append(fe.Function(V))
    h_nP1.append(fe.Function(V))
    mu_nP1.append(fe.Function(V))
    lam_nP1.append(fe.Function(V))

g_nM1 = []
h_nM1 = []
mu_nM1 = []
lam_nM1 = []
for idx in range(Q):
    g_nM1.append( fe.Function(V) ) 
    h_nM1.append( fe.Function(V) )
    mu_nM1.append( fe.Function(V) )
    lam_nM1.append( fe.Function(V) )
    
    
    

bilinear_forms_step2 = []
linear_forms_step1 = []
linear_forms_step2 = []

# Define variational problems for step 2 (CN timestep)

for idx in range(Q):
    bilinear_forms_step2.append( alpha_plus**2*g_trial[idx]*v*fe.dx\
        + alpha_plus*fe.dot( xi[idx], fe.grad(v) ) * g_trial[idx] * fe.dx\
            + alpha_plus*fe.dot( xi[idx], fe.grad(g_trial[idx]) )*v*fe.dx\
                + fe.dot( xi[idx], fe.grad(g_trial[idx]) )\
                    *fe.dot( xi[idx], fe.grad(v) )*fe.dx )

    body_force_np1 = body_Force_extrap(g_n, g_nM1, idx, Force_density)
    body_force_n = body_Force(vel(g_n), idx, Force_density)
    
    linear_forms_step1.append( ( alpha_minus*alpha_plus*g_n[idx]*v\
        + alpha_minus*g_n[idx]*fe.dot( xi[idx], fe.grad(v) )\
        +   (1/tau)*( g_equil_extrap(g_n, g_n, idx) + g_equil(g_n, idx) ) * alpha_plus*v\
        + (1/tau)*( g_equil_extrap(g_n, g_n, idx) + g_equil(g_n, idx) ) * fe.dot( xi[idx], fe.grad(v) )\
            - fe.dot( xi[idx], fe.grad(g_n[idx]) )*alpha_plus*v\
                - fe.dot( xi[idx], fe.grad(g_n[idx]) )*fe.dot( xi[idx], fe.grad(v) )\
                    + 0.5*(body_force_n + body_force_n)*alpha_plus*v\
                        + 0.5*(body_force_n + body_force_n)\
                            *fe.dot( xi[idx], fe.grad(v) ) )*fe.dx )

    linear_forms_step2.append( ( alpha_minus*alpha_plus*g_n[idx]*v\
        + alpha_minus*g_n[idx]*fe.dot( xi[idx], fe.grad(v) )\
        +   (1/tau)*( g_equil_extrap(g_n, g_nM1, idx) + g_equil(g_n, idx) ) * alpha_plus*v\
        + (1/tau)*( g_equil_extrap(g_n, g_nM1, idx) + g_equil(g_n, idx) ) * fe.dot( xi[idx], fe.grad(v) )\
            - fe.dot( xi[idx], fe.grad(g_n[idx]) )*alpha_plus*v\
                - fe.dot( xi[idx], fe.grad(g_n[idx]) )*fe.dot( xi[idx], fe.grad(v) )\
                    + 0.5*(body_force_np1 + body_force_n)*alpha_plus*v\
                        + 0.5*(body_force_np1 + body_force_n)\
                            *fe.dot( xi[idx], fe.grad(v) ) )*fe.dx )


# Assemble matrices for CN timestep
sys_mat_step2 = []
rhs_vec_step2 = [0]*Q
for idx in range(Q):
    sys_mat_step2.append( fe.assemble(bilinear_forms_step2[idx] ) )
    
    
# CN timestepping
t = 0
for n in range(0, num_steps):
    t += dt
    
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
        fe.solve( sys_mat_step2[idx], g_nP1[idx].vector(), rhs_vec_step2[idx] )
        
    # Update previous solutions
    for idx in range(Q):
        g_nM1[idx].assign( g_n[idx] )
    
    for idx in range(Q):
        g_n[idx].assign( g_nP1[idx] )
        
    fe.project(g_n[7], V, function=f5_lower_func)
    fe.project(g_n[4], V, function=f2_lower_func)
    fe.project(g_n[8], V, function=f6_lower_func)
    fe.project(g_n[5], V, function=f7_upper_func)
    fe.project(g_n[2], V, function=f4_upper_func)
    fe.project(g_n[6], V, function=f8_upper_func)
    
    if n%1000 == 0:
        u_expr = vel(g_n)
        V_vec = fe.VectorFunctionSpace(mesh, "P", 2, constrained_domain=pbc)
        u_n = fe.project(u_expr, V_vec)
        u_n_x = fe.project(u_n.split()[0], V)
        
        u_e = fe.Expression('u_max*( 1 - pow( (2*x[1]/L_y -1), 2 ) )',
                                     degree = 2, u_max = u_max, L_y = L_y)
        u_e = fe.interpolate(u_e, V)
        error = np.abs(u_e.vector().get_local() - u_n_x.vector().get_local()).max()
        print('t = %.4f: error = %.3g' % (t, error))
        print('max u:', u_n_x.vector().get_local().max())
        if n%10 == 0:
            error_vec.append(error)
            
    
error_vec = np.asarray(error_vec)
#%%
u_expr = vel(g_n)
V_vec = fe.VectorFunctionSpace(mesh, "P", 1, constrained_domain=pbc)
u = fe.project(u_expr, V_vec)


# Plot velocity field with larger arrows
# Plot velocity field with larger arrows
coords = V_vec.tabulate_dof_coordinates()[::2]  # Shape: (1056, 2)
u_values = u.vector().get_local().reshape((V_vec.dim() // 2, 2))  # Shape: (1056, 2)
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
plt.show()

#%%
plt.rcParams['text.usetex'] = True
# Plot velocity profile at x=L_x/2
num_points_analytical = 200
num_points_numerical = 10
y_values_analytical = np.linspace(0, L_y, num_points_analytical)
y_values_numerical = np.linspace(0, L_y, num_points_numerical)
x_fixed = L_x/2
points = [(x_fixed, y) for y in y_values_numerical]
u_x_values = []
u_ex = np.linspace(0, L_y, num_points_analytical)
nu = tau/3
u_max = Force_density[0]*L_y**2/(8*rho_init*nu)
for i in range(num_points_analytical):
    u_ex[i] = ( 1 - (2*y_values_analytical[i]/L_y -1)**2 )
    
for point in points:
    u_at_point = u(point)
    u_x_values.append(u_at_point[0] / u_max)
    
plt.figure()
plt.plot(y_values_numerical/L_y, u_x_values, 'o', label="FE soln.")
plt.plot(y_values_analytical/L_y, u_ex, label="Analytical soln.")
plt.ylabel(r"$u_x/u_{\mathrm{max}}$", fontsize=20)
plt.xlabel(r"$y/L_y$", fontsize=20)
title_str = f"Velocity profile at x = L_x/2, tau = {tau}"
#plt.title(title_str)
plt.legend()
plt.tick_params(direction="in")
plt.show()

#%% Create grid of u_x and u_y values

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


#%% Create 2D grids of each f_i at final time

# 1) Extract the coordinates of each degree of freedom in V
coords_f = V.tabulate_dof_coordinates().reshape(-1, 2)
x_f = coords_f[:, 0]
y_f = coords_f[:, 1]

# 2) Find unique levels and check grid size
x_unique = np.unique(x_f)
y_unique = np.unique(y_f)
nx_f = len(x_unique)
ny_f = len(y_unique)
assert nx_f * ny_f == x_f.size, "grid size mismatch for g_i"

# 3) Compute lexicographic ordering so that slow index=y, fast=x
order_f = np.lexsort((x_f, y_f))

# 4) Loop over all distributions, sort & reshape
g_grids = []
for idx, fi in enumerate(g_n):
    # flatten values, sort into (y,x) lex order, then reshape into (ny, nx)
    fi_vals   = fi.vector().get_local()
    fi_sorted = fi_vals[order_f]
    fi_grid   = fi_sorted.reshape((ny_f, nx_f))
    g_grids.append(fi_grid)
    # Optional: if you want to name them individually:
    # globals()[f"f{idx}_grid"] = fi_grid

# Now f_grids[i] is the (ny_f × nx_f) array of f_i values at the mesh grid.
# e.g., f_grids[0] is f0_grid, f_grids[1] is f1_grid, etc.

    


