import fenics as fe
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

T = 400
dt = 1
num_steps = int(np.ceil(T/dt))
tau = 1.0

nx = ny = 16
L_x = L_y = 32
h = L_x/nx

# Number of discrete velocities
Q = 9
Force_density = np.array([2.6041666e-5, 0.0])

#Force prefactor 
alpha_plus = ( 2/dt + 1/tau )
alpha_minus = ( 2/dt - 1/tau )

# Density on wall
rho_wall = 1.0
# Initial density 
rho_init = 1.0
u_wall = (0.0, 0.0)

# Lattice speed of sound
c_s = 1/np.sqrt(3)

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

mesh = fe.RectangleMesh(fe.Point(0,0), fe.Point(32, 32), nx, nx )

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
V_vec = fe.VectorFunctionSpace(mesh, "P", 1, constrained_domain=pbc)

# Define trial and test functions
f0, f1, f2 = fe.TrialFunction(V), fe.TrialFunction(V), fe.TrialFunction(V)
f3, f4, f5 = fe.TrialFunction(V), fe.TrialFunction(V), fe.TrialFunction(V)
f6, f7, f8 = fe.TrialFunction(V), fe.TrialFunction(V), fe.TrialFunction(V)

trial_fns = [f0, f1, f2, f3, f4, f5, f6, f7, f8]

v = fe.TestFunction(V)

# Define functions for solutions at previous time steps
# f0_n, f1_n, f2_n = fe.Function(V), fe.Function(V), fe.Function(V)
# f3_n, f4_n, f5_n = fe.Function(V), fe.Function(V), fe.Function(V)
# f6_n, f7_n, f8_n = fe.Function(V), fe.Function(V), fe.Function(V)

# f_n = [f0_n, f1_n, f2_n, f3_n, f4_n, f5_n, f6_n, f7_n, f8_n]


# Define density
def rho(f_list):
    return f_list[0] + f_list[1] + f_list[2] + f_list[3] + f_list[4]\
        + f_list[5] + f_list[6] + f_list[7] + f_list[8]

# Define velocity
def vel(f_list):
    distr_fn_sum = f_list[0]*xi[0] + f_list[1]*xi[1] + f_list[2]*xi[2]\
        + f_list[3]*xi[3] + f_list[4]*xi[4] + f_list[5]*xi[5]\
            + f_list[6]*xi[6] + f_list[7]*xi[7] + f_list[8]*xi[8]
            
    density = rho(f_list)
    
    vel_term1 = distr_fn_sum/density
    
    F = fe.Constant( (Force_density[0], Force_density[1]) )
    vel_term2 = F * dt / ( 2 * density )
    
    
    return vel_term1 + vel_term2


# Define initial equilibrium distributions
def f_equil_init(vel_idx, Force_density):
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
def f_equil(f_list, vel_idx):
    rho_expr = sum(fj for fj in f_list)
    u_expr   = vel(f_list)    
    ci       = xi[vel_idx]
    ci_dot_u = fe.dot(ci, u_expr)
    return w[vel_idx] * rho_expr * (
        1
        + ci_dot_u / c_s**2
        + ci_dot_u**2 / (2*c_s**4)
        - fe.dot(u_expr, u_expr) / (2*c_s**2)
    )

def f_equil_extrap(f_list_n, f_list_n_1, vel_idx):
    rho_expr = sum(fj for fj in f_list_n)
    u_expr   = vel(f_list_n)    
    ci       = xi[vel_idx]
    ci_dot_u = fe.dot(ci, u_expr)
    
    f_equil_n = w[vel_idx] * rho_expr * (
        1
        + ci_dot_u / c_s**2
        + ci_dot_u**2 / (2*c_s**4)
        - fe.dot(u_expr, u_expr) / (2*c_s**2)
    )
    
    rho_expr = sum(fj for fj in f_list_n_1)
    u_expr   = vel(f_list_n_1)   
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
    return -( f_list[vel_idx] - f_equil(f_list, vel_idx) ) / tau

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
    vel_n = vel(f_list_n)
    vel_n_1 = vel(f_list_n_1)
    
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
    
    


# Initialize distribution functions. We will use 
# f_i^{0} \gets f_i^{0, eq}( \rho_0, \bar{u}_0 ),
# where \bar{u}_0 = u_0 - F\Delta t/( 2 \rho_0 ).
# Here we will take u_0 = 0.

f0_n, f1_n, f2_n = fe.Function(V), fe.Function(V), fe.Function(V)
f3_n, f4_n, f5_n = fe.Function(V), fe.Function(V), fe.Function(V)
f6_n, f7_n, f8_n = fe.Function(V), fe.Function(V), fe.Function(V)

f0_n = fe.project(f_equil_init(0, Force_density), V )
f1_n = fe.project(f_equil_init(1, Force_density), V )
f2_n = fe.project(f_equil_init(2, Force_density), V )
f3_n = fe.project(f_equil_init(3, Force_density), V )
f4_n = fe.project(f_equil_init(4, Force_density), V )
f5_n = fe.project(f_equil_init(5, Force_density), V )
f6_n = fe.project(f_equil_init(6, Force_density), V )
f7_n = fe.project(f_equil_init(7, Force_density), V )
f8_n = fe.project(f_equil_init(8, Force_density), V )

f_n = [f0_n, f1_n, f2_n, f3_n, f4_n, f5_n, f6_n, f7_n, f8_n]

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
 
f5_lower = w[5] * fe.Constant(rho_wall) # rho_expr
f2_lower = w[2] * fe.Constant(rho_wall) # rho_expr 
f6_lower = w[6] * fe.Constant(rho_wall) # rho_expr

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
        if fe.near(x[1], 32, tol):
            return True
        else:
            return False
    else:
        return False

rho_expr = sum( fk for fk in f_n )
 
f7_upper = w[7] * fe.Constant(rho_wall) # rho_expr
f4_upper = w[4] * fe.Constant(rho_wall) # rho_expr 
f8_upper = w[8] * fe.Constant(rho_wall) # rho_expr

f7_upper_func = fe.Function(V)
f4_upper_func = fe.Function(V)
f8_upper_func = fe.Function(V)

fe.project( f7_upper, V, function=f7_upper_func )
fe.project( f4_upper, V, function=f4_upper_func )
fe.project( f8_upper, V, function=f8_upper_func )

bc_f7 = fe.DirichletBC(V, f7_upper_func, Bdy_Upper)
bc_f4 = fe.DirichletBC(V, f4_upper_func, Bdy_Upper)
bc_f8 = fe.DirichletBC(V, f8_upper_func, Bdy_Upper)


a_array_step1 = []
L_array_step1 = []
A_array_step1 = []
b_array_step1 = []

f_nP1 = []
f_nP1_vec = []

# Define variational problems

for idx in range(Q):
    a_array_step1.append( trial_fns[idx] * v * fe.dx\
                   + dt*fe.dot( xi[idx], fe.grad(trial_fns[idx]) ) * v * fe.dx )
    
    L_array_step1.append( ( f_n[idx] + dt*coll_op(f_n, idx)\
      + dt * body_Force( vel(f_n), idx, Force_density) ) * v * fe.dx )
        
    A_array_step1.append( fe.assemble(a_array_step1[idx]) )
    
    f_nP1.append( fe.Function(V) )
    f_nP1_vec.append( f_nP1[idx].vector() )


t = 0 
for n in range(1):
    # Update current time
    t += dt
    
    # Assemble right-hand side vectors
    
    for idx in range(Q):
        b_array_step1.append( fe.assemble(L_array_step1[idx]) )
    
    # Apply BCs for distribution functions 5, 2, and 6
    bc_f5.apply(A_array_step1[5], b_array_step1[5])
    bc_f2.apply(A_array_step1[2], b_array_step1[2])
    bc_f6.apply(A_array_step1[6], b_array_step1[6])
    
    # Apply BCs for distribution functions 7, 4, 8
    bc_f7.apply(A_array_step1[7], b_array_step1[7])
    bc_f4.apply(A_array_step1[4], b_array_step1[4])
    bc_f8.apply(A_array_step1[8], b_array_step1[8])
    
    for idx in range(Q):
        fe.solve(A_array_step1[idx], f_nP1_vec[idx], b_array_step1[idx])

    fe.project(w[5]*fe.Constant(rho_wall), V, function=f5_lower_func)
    fe.project(w[2]*fe.Constant(rho_wall), V, function=f2_lower_func)
    fe.project(w[6]*fe.Constant(rho_wall), V, function=f6_lower_func)
    fe.project(w[7]*fe.Constant(rho_wall), V, function=f7_upper_func)
    fe.project(w[4]*fe.Constant(rho_wall), V, function=f4_upper_func)
    fe.project(w[8]*fe.Constant(rho_wall), V, function=f8_upper_func)
    
    # Solve linear system in each time step

    
    
# We will do the explicit procedure for only one timestep.
# We then change bilinear and linear forms for CN-LS Galerkin
# We will now need two new variables fi_n_1 - ie f_i^{n-1}
# for use in the extrapolated collision operato


f_nM1 = []

# Assign initial values to fi_n_1.
for idx in range(Q):
    f_nM1.append(fe.Function(V) )
    f_nM1[idx].assign(f_n[idx])
    f_n[idx].assign(f_nP1[idx])

a_array_step2 = []
L_array_step2 = []
# Bilinear and linear terms for f0
for idx in range(Q):
    a_array_step2.append( alpha_plus**2*trial_fns[idx]*v*fe.dx\
        + alpha_plus*fe.dot( xi[0], fe.grad(v) ) * trial_fns[idx] * fe.dx\
            + alpha_plus*fe.dot( xi[0], fe.grad(trial_fns[idx]) )*v*fe.dx\
                + fe.dot( xi[0], fe.grad(trial_fns[idx]) )*fe.dot( xi[0], fe.grad(v) )*fe.dx )
    
    body_force_np1 = body_Force_extrap(f_n, f_nM1, idx, Force_density)
    body_force_n = body_Force(vel(f_n), idx, Force_density)
    
    L_array_step2.append( ( alpha_minus*alpha_plus*f0_n*v\
        + alpha_minus*f_n[idx]*fe.dot( xi[idx], fe.grad(v) )\
        +   (1/tau)*( f_equil_extrap(f_n, f_nM1, idx) + f_equil(f_n, idx) ) * alpha_plus*v\
        + (1/tau)*( f_equil_extrap(f_n, f_nM1, idx) + f_equil(f_n, idx) ) * fe.dot( xi[idx], fe.grad(v) )\
            - fe.dot( xi[idx], fe.grad(f_n[idx]) )*alpha_plus*v\
                - fe.dot( xi[idx], fe.grad(f_n[idx]) )*fe.dot( xi[idx], fe.grad(v) )\
                    + 0.5*(body_force_np1 + body_force_n)*alpha_plus*v\
                        + 0.5*(body_force_np1 + body_force_n)\
                            *fe.dot( xi[idx], fe.grad(v) ) )*fe.dx )


A_array_step2 = []
b_array_step2 = []

# Assemble matrices
for idx in range(Q):
    A_array_step2.append(fe.assemble(a_array_step2[idx]))
    
    
for n in range(1, num_steps):
    # Update current time
    t += dt
    
    for idx in range(Q):
        b_array_step2.append( fe.assemble(L_array_step2[idx]) )
    
    # Apply BCs for distribution functions 5, 2, and 6
    bc_f5.apply(A_array_step2[5], b_array_step2[5])
    bc_f2.apply(A_array_step2[2], b_array_step2[2])
    bc_f6.apply(A_array_step2[6], b_array_step2[6])
    
    # Apply BCs for distribution functions 7, 4, 8
    bc_f7.apply(A_array_step2[7], b_array_step2[7])
    bc_f4.apply(A_array_step2[4], b_array_step2[4])
    bc_f8.apply(A_array_step2[8], b_array_step2[8])
    
    for idx in range(Q):
        fe.solve(A_array_step1[idx], f_nP1_vec[idx], b_array_step1[idx])
    
    # Solve linear system in each time step
    
    # Update previous solution
    
    for idx in range(Q):
        f_nM1[idx].assign( f_n[idx] )
        f_n[idx].assign(f_nP1[idx])
    
    
    fe.project(w[5]*fe.Constant(rho_wall), V, function=f5_lower_func)
    fe.project(w[2]*fe.Constant(rho_wall), V, function=f2_lower_func)
    fe.project(w[6]*fe.Constant(rho_wall), V, function=f6_lower_func)
    fe.project(w[7]*fe.Constant(rho_wall), V, function=f7_upper_func)
    fe.project(w[4]*fe.Constant(rho_wall), V, function=f4_upper_func)
    fe.project(w[8]*fe.Constant(rho_wall), V, function=f8_upper_func)



u_expr = vel(f_n)
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
# Plot velocity profile at x=0.5 (unchanged, assuming it works)
num_points = 200
y_values = np.linspace(0, 32, num_points)
x_fixed = 0.5
points = [(x_fixed, y) for y in y_values]
u_x_values = []
u_ex = np.linspace(0, 32, num_points)
u_max = 0.01
for i in range(num_points):
    u_ex[i] = u_max*( 1 - (2*y_values[i]/L_x -1)**2 )
    
for point in points:
    u_at_point = u(point)
    u_x_values.append(u_at_point[0])
plt.figure()
plt.plot(u_x_values, y_values)
plt.plot(u_ex, y_values, 'o')
plt.xlabel("u_x")
plt.ylabel("y")
plt.title("Velocity profile at x=0.5")
plt.show()

#%% Create grid of u_x and u_y values

# figure out unique x- and y- levels
x_unique = np.unique(x)
y_unique = np.unique(y)
nx = len(x_unique)
ny = len(y_unique)
assert nx*ny == u_x.size, "grid size mismatch"

# now sort the flat arrays into lexicographic (y,x) order
# we want the slow index to be y, fast index x, so lexsort on (x,y)
order = np.lexsort((x, y))

# apply that ordering
u_x_sorted = u_x[order]
u_y_sorted = u_y[order]

# reshape into (ny, nx).  If your mesh is square, nx==ny.
u_x_grid = u_x_sorted.reshape((ny, nx))
u_y_grid = u_y_sorted.reshape((ny, nx))

