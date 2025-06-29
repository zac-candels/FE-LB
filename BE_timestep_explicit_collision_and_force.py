import fenics as fe
import numpy as np
import matplotlib.pyplot as plt


T = 10.0 
num_steps = 1000
dt = T / num_steps
tau = 1.0

# Number of discrete velocities
Q = 9
Force_density = np.array([2.6e-5, 0.0])

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
xi = np.array([
    [0,0], [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1]
])

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

f = [f0, f1, f2, f3, f4, f5, f6, f7, f8]

v = fe.TestFunction(V)

# Define functions for solutions at previous time steps
f0_n, f1_n, f2_n = fe.Function(V), fe.Function(V), fe.Function(V)
f3_n, f4_n, f5_n = fe.Function(V), fe.Function(V), fe.Function(V)
f6_n, f7_n, f8_n = fe.Function(V), fe.Function(V), fe.Function(V)

f_n = [f0_n, f1_n, f2_n, f3_n, f4_n, f5_n, f6_n, f7_n, f8_n]


# Define density
def rho(f):
    return f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8]

# Define velocity
def vel(f):
    velocity = f[0]*xi[0] + f[1]*xi[1] + f[2]*xi[2] + f[3]*xi[3] + f[4]*xi[4]\
        + f[5]*xi[5] + f[6]*xi[6] + f[7]*xi[7] + f[8]*xi[8]
    return velocity

# Define equilibrium distribution
def f_equil(vel_idx):
    prefactor = xi[vel_idx] * rho 
    U_dot_ci = np.dot( vel, xi[vel_idx] )
    second_term = U_dot_ci / c_s**2
    third_term = U_dot_ci**2 / ( 2 * c_s**4 )
    fourth_term = np.dot( vel, vel ) / (2 * c_s**2 )
    return prefactor * (1 + second_term + third_term - fourth_term)

# Define collision operator
def coll_op(f, vel_idx):
    return ( f[vel_idx] - f_equil(vel_idx) ) / tau

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

f5_lower = fe.Expression("rho*w_5", degree = 2, rho = rho(f), w_5 = w[5])
f2_lower = fe.Expression("rho*w_2", degree = 2, rho = rho(f), w_2 = w[2])
f6_lower = fe.Expression("rho*w_6", degree = 2, rho = rho(f), w_6 = w[6])

bc_f5 = fe.DirichletBC(V, f5_lower, Bdy_Lower)
bc_f2 = fe.DirichletBC(V, f2_lower, Bdy_Lower)
bc_f6 = fe.DirichletBC(V, f6_lower, Bdy_Lower)

# Similarly, we will define boundary conditions for f_7, f_4, and f_8
# at the upper wall. Once again, boundary conditions simply reduce
# to \rho * w_i

tol = 1e-14
def Bdy_Upper(x, on_boundary):
    if on_boundary:
        if fe.near(x[1], 1, tol):
            return True
        else:
            return False
    else:
        return False

f7_upper = fe.Expression("rho*w_7", degree = 2, rho = rho(f), w_7 = w[7])
f4_upper = fe.Expression("rho*w_4", degree = 2, rho = rho(f), w_4 = w[4])
f8_upper = fe.Expression("rho*w_8", degree = 2, rho = rho(f), w_8 = w[8])

bc_f7 = fe.DirichletBC(V, f7_upper, Bdy_Upper)
bc_f4 = fe.DirichletBC(V, f4_upper, Bdy_Upper)
bc_f8 = fe.DirichletBC(V, f8_upper, Bdy_Upper)


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

f_n = [f0_n, f1_n, f2_n, f3_n, f4_n, f5_n, f6_n, f7_n, f8_n]


# Define variational problems
a0 = f0 * v * fe.dx + dt*fe.dot( xi[0], fe.grad(f0) ) * v * fe.dx 
L0 = ( f0_n + dt*coll_op(f0_n, 0)\
      + dt * body_Force( vel(f_n), 0, Force_density) ) * v * fe.dx 

a1 = f0 * v * fe.dx + dt*fe.dot( xi[1], fe.grad(f1) ) * v * fe.dx 
L1 = ( f1_n + dt*coll_op(f1_n, 1)\
      + dt * body_Force( vel(f_n), 1, Force_density) ) * v * fe.dx 

a2 = f2 * v * fe.dx + dt*fe.dot( xi[2], fe.grad(f2) ) * v * fe.dx 
L2 = ( f2_n + dt*coll_op(f2_n, 2)\
      + dt * body_Force( vel(f_n), 2, Force_density) ) * v * fe.dx 

a3 = f3 * v * fe.dx + dt*fe.dot( xi[3], fe.grad(f3) ) * v * fe.dx 
L3 = ( f3_n + dt*coll_op(f3_n, 3)\
      + dt * body_Force( vel(f_n), 3, Force_density) ) * v * fe.dx  

a4 = f4 * v * fe.dx + dt*fe.dot( xi[4], fe.grad(f4) ) * v * fe.dx 
L4 = ( f4_n + dt*coll_op(f4_n, 4)\
      + dt * body_Force( vel(f_n), 4, Force_density) ) * v * fe.dx 

a5 = f5 * v * fe.dx + dt*fe.dot( xi[5], fe.grad(f5) ) * v * fe.dx 
L5 = ( f5_n + dt*coll_op(f5_n, 5)\
      + dt * body_Force( vel(f_n), 5, Force_density) ) * v * fe.dx 

a6 = f6 * v * fe.dx + dt*fe.dot( xi[6], fe.grad(f6) ) * v * fe.dx 
L6 = ( f6_n + dt*coll_op(f6_n, 6)\
      + dt * body_Force( vel(f_n), 6, Force_density) ) * v * fe.dx 

a7 = f7 * v * fe.dx + dt*fe.dot( xi[7], fe.grad(f7) ) * v * fe.dx 
L7 = ( f7_n + dt*coll_op(f7_n, 7)\
      + dt * body_Force( vel(f_n), 7, Force_density) ) * v * fe.dx  

a8 = f8 * v * fe.dx + dt*fe.dot( xi[8], fe.grad(f8) ) * v * fe.dx 
L8 = ( f8_n + dt*coll_op(f8_n, 8)\
      + dt * body_Force( vel(f_n), 8, Force_density) ) * v * fe.dx 

# Assemble matrices
A0, A1, A2 = fe.assemble(a0), fe.assemble(a1), fe.assemble(a3)
A3, A4, A5 = fe.assemble(a3), fe.assemble(a4), fe.assemble(a5)
A6, A7, A8 = fe.assemble(a6), fe.assemble(a7), fe.assemble(a8)

# Assemble right-hand side vectors
b0, b1, b2 = fe.assemble(L0), fe.assemble(L1), fe.assemble(L2)
b3, b4, b5 = fe.assemble(L3), fe.assemble(L4), fe.assemble(L5)
b6, b7, b8 = fe.assemble(L6), fe.assemble(L7), fe.assemble(L8)

# Time-stepping
f0, f1, f2 = fe.Function(V), fe.Function(V), fe.Function(V)
f3, f4, f5 = fe.Function(V), fe.Function(V), fe.Function(V)
f6, f7, f8 = fe.Function(V), fe.Function(V), fe.Function(V)
t = 0 
for n in range(num_steps):
    # Update current time
    t += dt
    
    
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
    
    










        
