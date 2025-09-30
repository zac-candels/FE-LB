import fenics as fe
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

T = 2000
dt = 1
num_steps = int(np.ceil(T/dt))


Re = 0.96
nx = ny = 5
L_x = 32
L_y = 32
h = L_x/nx

error_vec = []

# Lattice speed of sound
c_s = np.sqrt(1/3) # np.sqrt( 1./3. * h**2/dt**2 )

nu = 1.0/3.0
tau = 1 #nu/c_s**2 + 0.5*dt 

# Number of discrete velocities
Q = 9
Force_density = np.array([2.6041666e-5, 0.0])

# Density on wall
rho_wall = 1.0
# Initial density 
rho_init = 1.0
u_wall = (0.0, 0.0)


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

f_trial = fe.TrialFunction(V)
f_n = []
for idx in range(Q):
    f_n.append(fe.Function(V))
    
v = fe.TestFunction(V)

# Define FE functions to hold post-streaming solution at nP1 timesteps
f_nP1 = []
for idx in range(Q):
    f_nP1.append(fe.Function(V))
    
# Define FE functions to hold post-collision distributions
f_star = []
for idx in range(Q):
    f_star.append(fe.Function(V))  



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

# Define collision operator
def coll_op(f_list, vel_idx):
    return -( f_list[vel_idx] - f_equil(f_list, vel_idx) ) / (tau + 0.5)

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

bilinear_forms_stream = []
linear_forms_stream = []

bilinear_forms_collision = []
linear_forms_collision = []

n = fe.FacetNormal(mesh)
opp_idx = {0:0, 1:3, 2:4, 3:1, 4:2, 5:7, 6:8, 7:5, 8:6}

for idx in range(Q):
    
    f_eq = f_equil(f_n, idx)
    linear_forms_collision.append( ( f_n[idx]\
                                    - dt/(tau + 0.5) * (f_n[idx] - f_eq) )*v*fe.dx  )
    
    bilinear_forms_stream.append( f_trial * v * fe.dx )
    
    double_dot_product_term = -0.5*dt**2 * fe.dot( xi[idx], fe.grad(f_n[idx]) )\
        * fe.dot( xi[idx], fe.grad(v) ) * fe.dx
        
    dot_product_force_term = 0.5*dt**2 * fe.dot( xi[idx], fe.grad(v) )\
        * body_Force( vel(f_n), idx, Force_density ) * fe.dx
    
        
    lin_form_idx = f_n[idx]*v*fe.dx\
        - dt*v*fe.dot( xi[idx], fe.grad(f_n[idx]) )*fe.dx\
            + dt*v*body_Force( vel(f_n), idx, Force_density )*fe.dx\
               + double_dot_product_term\
                   + dot_product_force_term\
    
    linear_forms_stream.append( lin_form_idx )
        
# Assemble matrices for first step
sys_mat = []
rhs_vec_streaming = [0]*Q
rhs_vec_collision = [0]*Q
for idx in range(Q):
    sys_mat.append( fe.assemble( bilinear_forms_stream[idx] ) )
    
# Timestepping
t = 0.0
for n in range(num_steps):
    t += dt
    
    for idx in range(Q):
        rhs_vec_collision[idx] = fe.assemble( linear_forms_collision[idx] )
        
    for idx in range(Q):
        f_eq = f_equil(f_n, idx)
        f_n[idx] = fe.project( f_n[idx] - dt/(tau + 0.5) * (f_n[idx] - f_eq), V)
    
    # Assemble RHS vectors
    for idx in range(Q):
        rhs_vec_streaming[idx] = ( fe.assemble(linear_forms_stream[idx]) )
        
    fe.project(f_n[7], V, function=f5_lower_func)
    fe.project(f_n[4], V, function=f2_lower_func)
    fe.project(f_n[8], V, function=f6_lower_func)
    fe.project(f_n[5], V, function=f7_upper_func)
    fe.project(f_n[2], V, function=f4_upper_func)
    fe.project(f_n[6], V, function=f8_upper_func)
    
    # Apply BCs for distribution functions 5, 2, and 6
    bc_f5.apply(sys_mat[5], rhs_vec_streaming[5])
    bc_f2.apply(sys_mat[2], rhs_vec_streaming[2])
    bc_f6.apply(sys_mat[6], rhs_vec_streaming[6])
    
    # Apply BCs for distribution functions 7, 4, 8
    bc_f7.apply(sys_mat[7], rhs_vec_streaming[7])
    bc_f4.apply(sys_mat[4], rhs_vec_streaming[4])
    bc_f8.apply(sys_mat[8], rhs_vec_streaming[8])
    
    # Solve linear system in each timestep
    for idx in range(Q):
        fe.solve( sys_mat[idx], f_nP1[idx].vector(), rhs_vec_streaming[idx] )
        
    # Update previous solutions
    
    for idx in range(Q):
        f_n[idx].assign( f_nP1[idx] )
        
    
    if n%100 == 0:
        u_expr = vel(f_n)
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
u_expr = vel(f_n)
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
assert nx_f * ny_f == x_f.size, "grid size mismatch for f_i"

# 3) Compute lexicographic ordering so that slow index=y, fast=x
order_f = np.lexsort((x_f, y_f))

# 4) Loop over all distributions, sort & reshape
f_grids = []
for idx, fi in enumerate(f_n):
    # flatten values, sort into (y,x) lex order, then reshape into (ny, nx)
    fi_vals   = fi.vector().get_local()
    fi_sorted = fi_vals[order_f]
    fi_grid   = fi_sorted.reshape((ny_f, nx_f))
    f_grids.append(fi_grid)
    # Optional: if you want to name them individually:
    # globals()[f"f{idx}_grid"] = fi_grid

# Now f_grids[i] is the (ny_f × nx_f) array of f_i values at the mesh grid.
# e.g., f_grids[0] is f0_grid, f_grids[1] is f1_grid, etc.

    


