import fenics as fe
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

T = 10
dt = 0.015
num_steps = int(np.ceil(T/dt))
tau = 1.0

# Number of discrete velocities
Q = 9
Force_density = np.array([2.6e-4, 0.0])

#Force prefactor 
alpha = ( 2/dt + 1/tau )

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
nx = ny = 32
L_x = L_y = 1
mesh = fe.UnitSquareMesh(nx, ny)

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

f_list = [f0, f1, f2, f3, f4, f5, f6, f7, f8]

v = fe.TestFunction(V)

# Define functions for solutions at previous time steps
f0_n, f1_n, f2_n = fe.Function(V), fe.Function(V), fe.Function(V)
f3_n, f4_n, f5_n = fe.Function(V), fe.Function(V), fe.Function(V)
f6_n, f7_n, f8_n = fe.Function(V), fe.Function(V), fe.Function(V)

f_list_n = [f0_n, f1_n, f2_n, f3_n, f4_n, f5_n, f6_n, f7_n, f8_n]


# Define density
def rho(f_list):
    return f_list[0] + f_list[1] + f_list[2] + f_list[3] + f_list[4]\
        + f_list[5] + f_list[6] + f_list[7] + f_list[8]

# Define velocity
def vel(f_list):
    velocity = f_list[0]*xi[0] + f_list[1]*xi[1] + f_list[2]*xi[2]\
        + f_list[3]*xi[3] + f_list[4]*xi[4] + f_list[5]*xi[5]\
            + f_list[6]*xi[6] + f_list[7]*xi[7] + f_list[8]*xi[8]
    return velocity

# Define equilibrium distribution
def f_equil(f_list, vel_idx):
    rho_expr = sum(fj for fj in f_list)
    u_expr   = vel(f_list) / rho_expr    
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
    return -( f_list[vel_idx] - f_equil(f_list, vel_idx) ) / tau

def body_Force(vel, vel_idx, Force_density):
    prefactor = (1 - dt/( 2 * tau) )*w[vel_idx]
    inverse_cs2 = 1 / c_s**2
    inverse_cs4 = 1 / c_s**4
    
    first_term = ( xi[vel_idx][0]/ c_s**2\
                  + inverse_cs4 * ( xi[vel_idx][0]**2 - c_s**2 ) * vel[0] )\
        * Force_density[0]
        
    second_term = ( xi[vel_idx][1] / c_s**2\
                  + inverse_cs4 * xi[vel_idx][0] * xi[vel_idx][1] )\
        * Force_density[1]
        
    third_term = ( xi[vel_idx][0] / c_s**2\
                  + inverse_cs4 * xi[vel_idx][0] * xi[vel_idx][1] )\
        * Force_density[0]
        
    fourth_term = ( xi[vel_idx][1] / c_s**2\
                   + inverse_cs4 *( xi[vel_idx][1]**2 - c_s**2 ) * vel[1] )\
        * Force_density[1]
    
    return prefactor * ( first_term + second_term + third_term + fourth_term )


# Interpolate initial conditions. Since here we are taking 
# u(x, 0) \equiv 0, it is sufficient to have
# f_i(x, 0) = f_i^{eq}(rho, 0) = rho *w_i

f0_0 = fe.Expression("rho_init * w_0", degree = 2, 
                     rho_init = rho_init, w_0 = w[0])
f1_0 = fe.Expression("rho_init * w_1", degree = 2, 
                     rho_init = rho_init, w_1 = w[1])
f2_0 = fe.Expression("rho_init * w_2", degree = 2, 
                     rho_init = rho_init, w_2 = w[2])
f3_0 = fe.Expression("rho_init * w_3", degree = 2, 
                     rho_init = rho_init, w_3 = w[3])
f4_0 = fe.Expression("rho_init * w_4", degree = 2, 
                     rho_init = rho_init, w_4 = w[4])
f5_0 = fe.Expression("rho_init * w_5", degree = 2, 
                     rho_init = rho_init, w_5 = w[5])
f6_0 = fe.Expression("rho_init * w_6", degree = 2, 
                     rho_init = rho_init, w_6 = w[6])
f7_0 = fe.Expression("rho_init * w_7", degree = 2, 
                     rho_init = rho_init, w_7 = w[7])
f8_0 = fe.Expression("rho_init * w_8", degree = 2, 
                     rho_init = rho_init, w_8 = w[8])

f0_n, f1_n = fe.interpolate(f0_0, V), fe.interpolate(f1_0, V)
f2_n, f3_n = fe.interpolate(f2_0, V), fe.interpolate(f3_0, V)
f4_n, f5_n = fe.interpolate(f4_0, V), fe.interpolate(f5_0, V)
f6_n, f7_n = fe.interpolate(f6_0, V), fe.interpolate(f7_0, V)
f8_n = fe.interpolate(f8_0, V)

f_list_n = [f0_n, f1_n, f2_n, f3_n, f4_n, f5_n, f6_n, f7_n, f8_n]

# Define boundary conditions.

# For f_5, f_2, and f_6, equilibrium boundary conditions at lower wall
# Since we are applying equilibrium boundary conditions 
# and assuming no slip on solid walls, f_i^{eq} reduces to
# \rho * w_i

tol = 1e-14
def Bdy_Lower(x, on_boundary):
    if on_boundary:
        if fe.near(x[1], 0, tol):
            return True
        else:
            return False
    else:
        return False
    
rho_expr = sum( fk for fk in f_list_n )
 
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
        if fe.near(x[1], 1, tol):
            return True
        else:
            return False
    else:
        return False

rho_expr = sum( fk for fk in f_list_n )
 
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


# Define variational problems
a0 = f0 * v * fe.dx + dt*fe.dot( xi[0], fe.grad(f0) ) * v * fe.dx 
L0 = ( f0_n + dt*coll_op(f_list_n, 0)\
      + dt * body_Force( vel(f_list_n), 0, Force_density) ) * v * fe.dx 

a1 = f1 * v * fe.dx + dt*fe.dot( xi[1], fe.grad(f1) ) * v * fe.dx 
L1 = ( f1_n + dt*coll_op(f_list_n, 1)\
      + dt * body_Force( vel(f_list_n), 1, Force_density) ) * v * fe.dx 

a2 = f2 * v * fe.dx + dt*fe.dot( xi[2], fe.grad(f2) ) * v * fe.dx 
L2 = ( f2_n + dt*coll_op(f_list_n, 2)\
      + dt * body_Force( vel(f_list_n), 2, Force_density) ) * v * fe.dx 

a3 = f3 * v * fe.dx + dt*fe.dot( xi[3], fe.grad(f3) ) * v * fe.dx 
L3 = ( f3_n + dt*coll_op(f_list_n, 3)\
      + dt * body_Force( vel(f_list_n), 3, Force_density) ) * v * fe.dx  

a4 = f4 * v * fe.dx + dt*fe.dot( xi[4], fe.grad(f4) ) * v * fe.dx 
L4 = ( f4_n + dt*coll_op(f_list_n, 4)\
      + dt * body_Force( vel(f_list_n), 4, Force_density) ) * v * fe.dx 

a5 = f5 * v * fe.dx + dt*fe.dot( xi[5], fe.grad(f5) ) * v * fe.dx 
L5 = ( f5_n + dt*coll_op(f_list_n, 5)\
      + dt * body_Force( vel(f_list_n), 5, Force_density) ) * v * fe.dx 

a6 = f6 * v * fe.dx + dt*fe.dot( xi[6], fe.grad(f6) ) * v * fe.dx 
L6 = ( f6_n + dt*coll_op(f_list_n, 6)\
      + dt * body_Force( vel(f_list_n), 6, Force_density) ) * v * fe.dx 

a7 = f7 * v * fe.dx + dt*fe.dot( xi[7], fe.grad(f7) ) * v * fe.dx 
L7 = ( f7_n + dt*coll_op(f_list_n, 7)\
      + dt * body_Force( vel(f_list_n), 7, Force_density) ) * v * fe.dx  

a8 = f8 * v * fe.dx + dt*fe.dot( xi[8], fe.grad(f8) ) * v * fe.dx 
L8 = ( f8_n + dt*coll_op(f_list_n, 8)\
      + dt * body_Force( vel(f_list_n), 8, Force_density) ) * v * fe.dx 

# Assemble matrices
A0, A1, A2 = fe.assemble(a0), fe.assemble(a1), fe.assemble(a2)
A3, A4, A5 = fe.assemble(a3), fe.assemble(a4), fe.assemble(a5)
A6, A7, A8 = fe.assemble(a6), fe.assemble(a7), fe.assemble(a8)

# Time-stepping
f0, f1, f2 = fe.Function(V), fe.Function(V), fe.Function(V)
f3, f4, f5 = fe.Function(V), fe.Function(V), fe.Function(V)
f6, f7, f8 = fe.Function(V), fe.Function(V), fe.Function(V)
t = 0 
for n in range(num_steps):
    # Update current time
    t += dt
    
    # Assemble right-hand side vectors
    b0, b1, b2 = fe.assemble(L0), fe.assemble(L1), fe.assemble(L2)
    b3, b4, b5 = fe.assemble(L3), fe.assemble(L4), fe.assemble(L5)
    b6, b7, b8 = fe.assemble(L6), fe.assemble(L7), fe.assemble(L8)
    
    # Apply BCs for distribution functions 5, 2, and 6
    bc_f5.apply(A5, b5)
    bc_f2.apply(A2, b2)
    bc_f6.apply(A6, b6)
    
    # Apply BCs for distribution functions 7, 4, 8
    bc_f7.apply(A7, b7)
    bc_f4.apply(A4, b4)
    bc_f8.apply(A8, b8)
    
    f0Vec, f1Vec, f2Vec = f0.vector(), f1.vector(), f2.vector()
    f3Vec, f4Vec, f5Vec = f3.vector(), f4.vector(), f5.vector()
    f6Vec, f7Vec, f8Vec = f6.vector(), f7.vector(), f8.vector()
    
    fe.solve(A0, f0Vec, b0)
    fe.solve(A1, f1Vec, b1)
    fe.solve(A2, f2Vec, b2)
    fe.solve(A3, f3Vec, b3)
    fe.solve(A4, f4Vec, b4)
    fe.solve(A5, f5Vec, b5)
    fe.solve(A6, f6Vec, b6)
    fe.solve(A7, f7Vec, b7)
    fe.solve(A8, f8Vec, b8)
    
    # Solve linear system in each time step
    
    # Update previous solution
    f0_n.assign(f0)
    f1_n.assign(f1)
    f2_n.assign(f2)
    f3_n.assign(f3)
    f4_n.assign(f4)
    f5_n.assign(f5)
    f6_n.assign(f6)
    f7_n.assign(f7)
    f8_n.assign(f8)
    
    fe.project(w[5]*fe.Constant(rho_wall), V, function=f5_lower_func)
    fe.project(w[2]*fe.Constant(rho_wall), V, function=f2_lower_func)
    fe.project(w[6]*fe.Constant(rho_wall), V, function=f6_lower_func)
    fe.project(w[7]*fe.Constant(rho_wall), V, function=f7_upper_func)
    fe.project(w[4]*fe.Constant(rho_wall), V, function=f4_upper_func)
    fe.project(w[8]*fe.Constant(rho_wall), V, function=f8_upper_func)
    

u_expr = vel(f_list_n) / rho(f_list_n)
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

# Plot velocity profile at x=0.5 (unchanged, assuming it works)
num_points = 100
y_values = np.linspace(0, 1, num_points)
x_fixed = 0.5
points = [(x_fixed, y) for y in y_values]
u_x_values = []
for point in points:
    u_at_point = u(point)
    u_x_values.append(u_at_point[0])
plt.figure()
plt.plot(u_x_values, y_values)
plt.xlabel("u_x")
plt.ylabel("y")
plt.title("Velocity profile at x=0.5")
plt.show()