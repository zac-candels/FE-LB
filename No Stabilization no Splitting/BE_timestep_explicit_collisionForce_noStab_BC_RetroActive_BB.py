import fenics as fe
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

T = 100
dt = 0.01
num_steps = int(np.ceil(T/dt))
tau = 1.0

# Number of discrete velocities
Q = 9
Force_density = np.array([1e-5, 0.0])

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
nx = ny = 20
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
    return -( f_list[vel_idx] - f_equil(f_list, vel_idx) ) / tau

def body_Force(vel, vel_idx, Force_density):
    prefactor = (1 - dt/( 2 * tau) )*w[vel_idx]
    inverse_cs2 = 1 / c_s**2
    inverse_cs4 = 1 / c_s**4
    
    xi_dot_prod_F = xi[vel_idx][0]*Force_density[0]\
        + xi[vel_idx][1]*Force_density[1]
        
    u_dot_prod_F = vel[0]*Force_density[0] + vel[1]*Force_density[1]
    
    xi_dot_u = xi[vel_idx][0]*vel[0] + xi[vel_idx][1]*vel[1]
    
    Force = prefactor*( inverse_cs2*(xi_dot_prod_F - u_dot_prod_F)\
                       + inverse_cs4*xi_dot_u*xi_dot_prod_F)
        
    return Force


# Initialize distribution functions. We will use 
# f_i^{0} \gets f_i^{0, eq}( \rho_0, \bar{u}_0 ),
# where \bar{u}_0 = u_0 - F\Delta t/( 2 \rho_0 ).
# Here we will take u_0 = 0.

# f0_n, f1_n, f2_n = fe.Function(V), fe.Function(V), fe.Function(V)
# f3_n, f4_n, f5_n = fe.Function(V), fe.Function(V), fe.Function(V)
# f6_n, f7_n, f8_n = fe.Function(V), fe.Function(V), fe.Function(V)

f0_n = fe.project(f_equil_init(0, Force_density), V )
f1_n = fe.project(f_equil_init(1, Force_density), V )
f2_n = fe.project(f_equil_init(2, Force_density), V )
f3_n = fe.project(f_equil_init(3, Force_density), V )
f4_n = fe.project(f_equil_init(4, Force_density), V )
f5_n = fe.project(f_equil_init(5, Force_density), V )
f6_n = fe.project(f_equil_init(6, Force_density), V )
f7_n = fe.project(f_equil_init(7, Force_density), V )
f8_n = fe.project(f_equil_init(8, Force_density), V )

f_list_n = [f0_n, f1_n, f2_n, f3_n, f4_n, f5_n, f6_n, f7_n, f8_n]

# Precompute boundary DOFs
tol = 1e-8
dof_coords = V.tabulate_dof_coordinates()
dofs_lower = []
dofs_upper = []
for i, coord in enumerate(dof_coords):
    if abs(coord[1] - 0.0) < tol:  # Lower wall
        dofs_lower.append(i)
    if abs(coord[1] - 1.0) < tol:  # Upper wall
        dofs_upper.append(i)


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
    
    
    f0Vec, f1Vec, f2Vec = f0.vector(), f1.vector(), f2.vector()
    f3Vec, f4Vec, f5Vec = f3.vector(), f4.vector(), f5.vector()
    f6Vec, f7Vec, f8Vec = f6.vector(), f7.vector(), f8.vector()
    
    # Solve linear system in each time step
    fe.solve(A0, f0Vec, b0)
    fe.solve(A1, f1Vec, b1)
    fe.solve(A2, f2Vec, b2)
    fe.solve(A3, f3Vec, b3)
    fe.solve(A4, f4Vec, b4)
    fe.solve(A5, f5Vec, b5)
    fe.solve(A6, f6Vec, b6)
    fe.solve(A7, f7Vec, b7)
    fe.solve(A8, f8Vec, b8)
    
    # Post-processing: impose bounce-back on the boundaries
    f2_vec = f2.vector()
    f4_vec = f4.vector()
    f5_vec = f5.vector()
    f6_vec = f6.vector()
    f7_vec = f7.vector()
    f8_vec = f8.vector()
    
    # Lower wall (y = 0): Incoming populations = f2, f5, f6
    for dof in dofs_lower:
        f2_vec[dof] = f4_vec[dof] # f2 <- f4
        f5_vec[dof] = f7_vec[dof] # f5 <- f7
        f6_vec[dof] = f8_vec[dof] # f6 <- f8
        
    # Upper wall (y=1): Incoming = f4, f7, f8
    for dof in dofs_upper:
        f4_vec[dof] = f2_vec[dof]  # f4 = f2
        f7_vec[dof] = f5_vec[dof]  # f7 = f5
        f8_vec[dof] = f6_vec[dof]  # f8 = f6
    
    
    
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
    

u_expr = vel(f_list_n)
u = fe.project(u_expr, V_vec)

# Plot velocity field 
coords = V_vec.tabulate_dof_coordinates()[::2] 
u_values = u.vector().get_local().reshape((V_vec.dim() // 2, 2)) 
x = coords[:, 0]  # x-coordinates
y = coords[:, 1]  # y-coordinates
u_x = u_values[:, 0]  # x-components of velocity
u_y = u_values[:, 1]  # y-components of velocity

# Define arrow scale based on maximum velocity
max_u = np.max(np.sqrt(u_x**2 + u_y**2))
arrow_length = 0.05  # 5% of domain size
scale = max_u / arrow_length if max_u > 0 else 1

# Plot vector field
plt.figure()
M = np.hypot(u_x, u_y)
plt.quiver(x, y, u_x, u_y, M, scale=scale, scale_units='height')
plt.title("Velocity field at final time")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Plot velocity profile at midpoint of channel
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