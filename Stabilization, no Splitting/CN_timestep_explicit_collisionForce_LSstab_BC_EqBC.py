import fenics as fe
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

T = 6000
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

f_list_n = [f0_n, f1_n, f2_n, f3_n, f4_n, f5_n, f6_n, f7_n, f8_n]

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
    
rho_expr = sum( fk for fk in f_list_n )
 
f0_lower = w[0] * fe.Constant(rho_wall)
f1_lower = w[1] * fe.Constant(rho_wall)
f2_lower = w[2] * fe.Constant(rho_wall) # rho_expr 
f3_lower = w[3] * fe.Constant(rho_wall)
f4_lower = w[4] * fe.Constant(rho_wall)
f5_lower = w[5] * fe.Constant(rho_wall) # rho_expr
f6_lower = w[6] * fe.Constant(rho_wall) # rho_expr
f7_lower = w[7] * fe.Constant(rho_wall)
f8_lower = w[8] * fe.Constant(rho_wall)

f0_lower_func = fe.Function(V)
f1_lower_func = fe.Function(V)
f2_lower_func = fe.Function(V)
f3_lower_func = fe.Function(V)
f4_lower_func = fe.Function(V)
f5_lower_func = fe.Function(V)
f6_lower_func = fe.Function(V)
f7_lower_func = fe.Function(V)
f8_lower_func = fe.Function(V)

fe.project( f0_lower, V, function=f0_lower_func )
fe.project( f1_lower, V, function=f1_lower_func )
fe.project( f2_lower, V, function=f2_lower_func )
fe.project( f3_lower, V, function=f3_lower_func )
fe.project( f4_lower, V, function=f4_lower_func )
fe.project( f5_lower, V, function=f5_lower_func )
fe.project( f6_lower, V, function=f6_lower_func )
fe.project( f7_lower, V, function=f7_lower_func )
fe.project( f8_lower, V, function=f8_lower_func )

L_bc_f0 = fe.DirichletBC(V, f0_lower_func, Bdy_Lower)
L_bc_f1 = fe.DirichletBC(V, f1_lower_func, Bdy_Lower)
L_bc_f2 = fe.DirichletBC(V, f2_lower_func, Bdy_Lower)
L_bc_f3 = fe.DirichletBC(V, f3_lower_func, Bdy_Lower)
L_bc_f4 = fe.DirichletBC(V, f4_lower_func, Bdy_Lower)
L_bc_f5 = fe.DirichletBC(V, f5_lower_func, Bdy_Lower)
L_bc_f6 = fe.DirichletBC(V, f6_lower_func, Bdy_Lower)
L_bc_f7 = fe.DirichletBC(V, f7_lower_func, Bdy_Lower)
L_bc_f8 = fe.DirichletBC(V, f8_lower_func, Bdy_Lower)

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

rho_expr = sum( fk for fk in f_list_n )
 
f0_upper = w[0] * fe.Constant(rho_wall)
f1_upper = w[1] * fe.Constant(rho_wall)
f2_upper = w[2] * fe.Constant(rho_wall) # rho_expr 
f3_upper = w[3] * fe.Constant(rho_wall)
f4_upper = w[4] * fe.Constant(rho_wall)
f5_upper = w[5] * fe.Constant(rho_wall) # rho_expr
f6_upper = w[6] * fe.Constant(rho_wall) # rho_expr
f7_upper = w[7] * fe.Constant(rho_wall)
f8_upper = w[8] * fe.Constant(rho_wall)

f0_upper_func = fe.Function(V)
f1_upper_func = fe.Function(V)
f2_upper_func = fe.Function(V)
f3_upper_func = fe.Function(V)
f4_upper_func = fe.Function(V)
f5_upper_func = fe.Function(V)
f6_upper_func = fe.Function(V)
f7_upper_func = fe.Function(V)
f8_upper_func = fe.Function(V)

fe.project( f0_upper, V, function=f0_upper_func )
fe.project( f1_upper, V, function=f1_upper_func )
fe.project( f2_upper, V, function=f2_upper_func )
fe.project( f3_upper, V, function=f3_upper_func )
fe.project( f4_upper, V, function=f4_upper_func )
fe.project( f5_upper, V, function=f5_upper_func )
fe.project( f6_upper, V, function=f6_upper_func )
fe.project( f7_upper, V, function=f7_upper_func )
fe.project( f8_upper, V, function=f8_upper_func )

U_bc_f0 = fe.DirichletBC(V, f0_upper_func, Bdy_Upper)
U_bc_f1 = fe.DirichletBC(V, f1_upper_func, Bdy_Upper)
U_bc_f2 = fe.DirichletBC(V, f2_upper_func, Bdy_Upper)
U_bc_f3 = fe.DirichletBC(V, f3_upper_func, Bdy_Upper)
U_bc_f4 = fe.DirichletBC(V, f4_upper_func, Bdy_Upper)
U_bc_f5 = fe.DirichletBC(V, f5_upper_func, Bdy_Upper)
U_bc_f6 = fe.DirichletBC(V, f6_upper_func, Bdy_Upper)
U_bc_f7 = fe.DirichletBC(V, f7_upper_func, Bdy_Upper)
U_bc_f8 = fe.DirichletBC(V, f8_upper_func, Bdy_Upper)


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
for n in range(1):
    # Update current time
    t += dt
    
    # Assemble right-hand side vectors
    b0, b1, b2 = fe.assemble(L0), fe.assemble(L1), fe.assemble(L2)
    b3, b4, b5 = fe.assemble(L3), fe.assemble(L4), fe.assemble(L5)
    b6, b7, b8 = fe.assemble(L6), fe.assemble(L7), fe.assemble(L8)
    
    # Apply BCs for distribution functions 5, 2, and 6
    L_bc_f0.apply(A0, b0)
    L_bc_f1.apply(A1, b1)
    L_bc_f2.apply(A2, b2)
    L_bc_f3.apply(A3, b3)
    L_bc_f4.apply(A4, b4)
    L_bc_f5.apply(A5, b5)
    L_bc_f6.apply(A6, b6)
    L_bc_f7.apply(A7, b7)
    L_bc_f8.apply(A8, b8)
    
    # Apply BCs for distribution functions 7, 4, 8
    U_bc_f0.apply(A0, b0)
    U_bc_f1.apply(A1, b1)
    U_bc_f2.apply(A2, b2)
    U_bc_f3.apply(A3, b3)
    U_bc_f4.apply(A4, b4)
    U_bc_f5.apply(A5, b5)
    U_bc_f6.apply(A6, b6)
    U_bc_f7.apply(A7, b7)
    U_bc_f8.apply(A8, b8)
    
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
    
    fe.project(w[0]*fe.Constant(rho_wall), V, function=f0_lower_func)
    fe.project(w[1]*fe.Constant(rho_wall), V, function=f1_lower_func)
    fe.project(w[2]*fe.Constant(rho_wall), V, function=f2_lower_func)
    fe.project(w[3]*fe.Constant(rho_wall), V, function=f3_lower_func)
    fe.project(w[4]*fe.Constant(rho_wall), V, function=f4_lower_func)
    fe.project(w[5]*fe.Constant(rho_wall), V, function=f5_lower_func)
    fe.project(w[6]*fe.Constant(rho_wall), V, function=f6_lower_func)
    fe.project(w[7]*fe.Constant(rho_wall), V, function=f7_lower_func)
    fe.project(w[8]*fe.Constant(rho_wall), V, function=f8_lower_func)
    
    fe.project(w[0]*fe.Constant(rho_wall), V, function=f0_upper_func)
    fe.project(w[1]*fe.Constant(rho_wall), V, function=f1_upper_func)
    fe.project(w[2]*fe.Constant(rho_wall), V, function=f2_upper_func)
    fe.project(w[3]*fe.Constant(rho_wall), V, function=f3_upper_func)
    fe.project(w[4]*fe.Constant(rho_wall), V, function=f4_upper_func)
    fe.project(w[5]*fe.Constant(rho_wall), V, function=f5_upper_func)
    fe.project(w[6]*fe.Constant(rho_wall), V, function=f6_upper_func)
    fe.project(w[7]*fe.Constant(rho_wall), V, function=f7_upper_func)
    fe.project(w[8]*fe.Constant(rho_wall), V, function=f8_upper_func)
    
    # Solve linear system in each time step
    
    
# We will do the explicit procedure for only one timestep.
# We then change bilinear and linear forms for CN-LS Galerkin
# We will now need two new variables fi_n_1 - ie f_i^{n-1}
# for use in the extrapolated collision operato

f0_n_1, f1_n_1, f2_n_1 = fe.Function(V), fe.Function(V), fe.Function(V)
f3_n_1, f4_n_1, f5_n_1 = fe.Function(V), fe.Function(V), fe.Function(V)
f6_n_1, f7_n_1, f8_n_1 = fe.Function(V), fe.Function(V), fe.Function(V)

# Assign initial values to fi_n_1.
f0_n_1.assign(f0_n)
f1_n_1.assign(f1_n)
f2_n_1.assign(f2_n)
f3_n_1.assign(f3_n)
f4_n_1.assign(f4_n)
f5_n_1.assign(f5_n)
f6_n_1.assign(f6_n)
f7_n_1.assign(f7_n)
f8_n_1.assign(f8_n)

# Update values from the previous timestep 
f0_n.assign(f0)
f1_n.assign(f1)
f2_n.assign(f2)
f3_n.assign(f3)
f4_n.assign(f4)
f5_n.assign(f5)
f6_n.assign(f6)
f7_n.assign(f7)
f8_n.assign(f8)

f_list_n_1 = [f0_n_1, f1_n_1, f2_n_1, f3_n_1, f4_n_1, f5_n_1,
              f6_n_1, f7_n_1, f8_n_1]

# Redefine f0,...,fn as trial functions
f0_trial, f1_trial, f2_trial = fe.TrialFunction(V), fe.TrialFunction(V), fe.TrialFunction(V)
f3_trial, f4_trial, f5_trial = fe.TrialFunction(V), fe.TrialFunction(V), fe.TrialFunction(V)
f6_trial, f7_trial, f8_trial = fe.TrialFunction(V), fe.TrialFunction(V), fe.TrialFunction(V)

# Bilinear and linear terms for f0
a0 = alpha_plus**2*f0_trial*v*fe.dx\
    + alpha_plus*fe.dot( xi[0], fe.grad(v) ) * f0_trial * fe.dx\
        + alpha_plus*fe.dot( xi[0], fe.grad(f0_trial) )*v*fe.dx\
            + fe.dot( xi[0], fe.grad(f0_trial) )*fe.dot( xi[0], fe.grad(v) )*fe.dx

body_force_np1 = body_Force_extrap(f_list_n, f_list_n_1, 0, Force_density)
body_force_n = body_Force(vel(f_list_n), 0, Force_density)

L0 = ( alpha_minus*alpha_plus*f0_n*v\
    + alpha_minus*f0_n*fe.dot( xi[0], fe.grad(v) )\
    +   (1/tau)*( f_equil_extrap(f_list_n, f_list_n_1, 0) + f_equil(f_list_n, 0) ) * alpha_plus*v\
    + (1/tau)*( f_equil_extrap(f_list_n, f_list_n_1, 0) + f_equil(f_list_n, 0) ) * fe.dot( xi[0], fe.grad(v) )\
        - fe.dot( xi[0], fe.grad(f0_n) )*alpha_plus*v\
            - fe.dot( xi[0], fe.grad(f0_n) )*fe.dot( xi[0], fe.grad(v) )\
                + 0.5*(body_force_np1 + body_force_n)*alpha_plus*v\
                    + 0.5*(body_force_np1 + body_force_n)\
                        *fe.dot( xi[0], fe.grad(v) ) )*fe.dx
 
# (Bi)linear forms for f1
a1 = alpha_plus**2*f1_trial*v*fe.dx\
    + alpha_plus*fe.dot( xi[1], fe.grad(v) ) * f1_trial * fe.dx\
        + alpha_plus*fe.dot( xi[1], fe.grad(f1_trial) )*v*fe.dx\
            + fe.dot( xi[1], fe.grad(f1_trial) ) * fe.dot( xi[1], fe.grad(v) )*fe.dx

body_force_np1 = body_Force_extrap(f_list_n, f_list_n_1, 1, Force_density)
body_force_n = body_Force(vel(f_list_n), 1, Force_density)

L1 = ( alpha_minus*alpha_plus*f1_n*v\
    + alpha_minus*f1_n*fe.dot( xi[1], fe.grad(v) )\
    +   (1/tau)*( f_equil_extrap(f_list_n, f_list_n_1, 1) + f_equil(f_list_n, 1) ) * alpha_plus*v\
    + (1/tau)*( f_equil_extrap(f_list_n, f_list_n_1, 1) + f_equil(f_list_n, 1) ) * fe.dot( xi[1], fe.grad(v) )\
        - fe.dot( xi[1], fe.grad(f1_n) )*alpha_plus*v\
            - fe.dot( xi[1], fe.grad(f1_n) )*fe.dot( xi[1], fe.grad(v) )\
                + 0.5*(body_force_np1 + body_force_n)*alpha_plus*v\
                    + 0.5*(body_force_np1 + body_force_n)\
                        *fe.dot( xi[1], fe.grad(v) ) )*fe.dx
  

# (Bi)linear forms for f2
a2 = alpha_plus**2*f2_trial*v*fe.dx\
    + alpha_plus*fe.dot( xi[2], fe.grad(v) ) * f2_trial * fe.dx\
        + alpha_plus*fe.dot( xi[2], fe.grad(f2_trial) )*v*fe.dx\
            + fe.dot( xi[2], fe.grad(f2_trial) ) * fe.dot( xi[2], fe.grad(v) )*fe.dx

body_force_np1 = body_Force_extrap(f_list_n, f_list_n_1, 2, Force_density)
body_force_n = body_Force(vel(f_list_n), 2, Force_density)

L2 = ( alpha_minus*alpha_plus*f2_n*v\
    + alpha_minus*f2_n*fe.dot( xi[2], fe.grad(v) )\
    +   (1/tau)*( f_equil_extrap(f_list_n, f_list_n_1, 2) + f_equil(f_list_n, 2) ) * alpha_plus*v\
    + (1/tau)*( f_equil_extrap(f_list_n, f_list_n_1, 2) + f_equil(f_list_n, 2) ) * fe.dot( xi[2], fe.grad(v) )\
        - fe.dot( xi[2], fe.grad(f2_n) )*alpha_plus*v\
            - fe.dot( xi[2], fe.grad(f2_n) )*fe.dot( xi[2], fe.grad(v) )\
                + 0.5*(body_force_np1 + body_force_n)*alpha_plus*v\
                    + 0.5*(body_force_np1 + body_force_n)\
                        *fe.dot( xi[2], fe.grad(v) ) )*fe.dx
    
    
# (Bi)linear forms for f3
a3 = alpha_plus**2*f3_trial*v*fe.dx\
    + alpha_plus*fe.dot( xi[3], fe.grad(v) ) * f3_trial * fe.dx\
        + alpha_plus*fe.dot( xi[3], fe.grad(f3_trial) )*v*fe.dx\
            + fe.dot( xi[3], fe.grad(f3_trial) ) * fe.dot( xi[3], fe.grad(v) )*fe.dx 

body_force_np1 = body_Force_extrap(f_list_n, f_list_n_1, 3, Force_density)
body_force_n = body_Force(vel(f_list_n), 3, Force_density)

L3 = ( alpha_minus*alpha_plus*f3_n*v\
    + alpha_minus*f3_n*fe.dot( xi[3], fe.grad(v) )\
    +   (1/tau)*( f_equil_extrap(f_list_n, f_list_n_1, 3) + f_equil(f_list_n, 3) ) * alpha_plus*v\
    + (1/tau)*( f_equil_extrap(f_list_n, f_list_n_1, 3) + f_equil(f_list_n, 3) ) * fe.dot( xi[3], fe.grad(v) )\
        - fe.dot( xi[3], fe.grad(f3_n) )*alpha_plus*v\
            - fe.dot( xi[3], fe.grad(f3_n) )*fe.dot( xi[3], fe.grad(v) )\
                + 0.5*(body_force_np1 + body_force_n)*alpha_plus*v\
                    + 0.5*(body_force_np1 + body_force_n)\
                        *fe.dot( xi[3], fe.grad(v) ) )*fe.dx
    
    
# (Bi)linear forms for f4
a4 = alpha_plus**2*f4_trial*v*fe.dx\
    + alpha_plus*fe.dot( xi[4], fe.grad(v) ) * f4_trial * fe.dx\
        + alpha_plus*fe.dot( xi[4], fe.grad(f4_trial) )*v*fe.dx\
            + fe.dot( xi[4], fe.grad(f4_trial) ) * fe.dot( xi[4], fe.grad(v) )*fe.dx 

body_force_np1 = body_Force_extrap(f_list_n, f_list_n_1, 4, Force_density)
body_force_n = body_Force(vel(f_list_n), 4, Force_density)

L4 = ( alpha_minus*alpha_plus*f4_n*v\
    + alpha_minus*f4_n*fe.dot( xi[4], fe.grad(v) )\
    +   (1/tau)*( f_equil_extrap(f_list_n, f_list_n_1, 4) + f_equil(f_list_n, 4) ) * alpha_plus*v\
    + (1/tau)*( f_equil_extrap(f_list_n, f_list_n_1, 4) + f_equil(f_list_n, 4) ) * fe.dot( xi[4], fe.grad(v) )\
        - fe.dot( xi[4], fe.grad(f4_n) )*alpha_plus*v\
            - fe.dot( xi[4], fe.grad(f4_n) )*fe.dot( xi[4], fe.grad(v) )\
                + 0.5*(body_force_np1 + body_force_n)*alpha_plus*v\
                    + 0.5*(body_force_np1 + body_force_n)\
                        *fe.dot( xi[4], fe.grad(v) ) )*fe.dx
    
    
# (Bi)linear forms for f5
a5 = alpha_plus**2*f5_trial*v*fe.dx\
    + alpha_plus*fe.dot( xi[5], fe.grad(v) ) * f5_trial * fe.dx\
        + alpha_plus*fe.dot( xi[5], fe.grad(f5_trial) )*v*fe.dx\
            + fe.dot( xi[5], fe.grad(f5_trial) ) * fe.dot( xi[5], fe.grad(v) )*fe.dx

body_force_np1 = body_Force_extrap(f_list_n, f_list_n_1, 5, Force_density)
body_force_n = body_Force(vel(f_list_n), 5, Force_density)

L5 = ( alpha_minus*alpha_plus*f5_n*v\
    + alpha_minus*f5_n*fe.dot( xi[5], fe.grad(v) )\
    +   (1/tau)*( f_equil_extrap(f_list_n, f_list_n_1, 5) + f_equil(f_list_n, 5) ) * alpha_plus*v\
    + (1/tau)*( f_equil_extrap(f_list_n, f_list_n_1, 5) + f_equil(f_list_n, 5) ) * fe.dot( xi[5], fe.grad(v) )\
        - fe.dot( xi[5], fe.grad(f5_n) )*alpha_plus*v\
            - fe.dot( xi[5], fe.grad(f5_n) )*fe.dot( xi[5], fe.grad(v) )\
                + 0.5*(body_force_np1 + body_force_n)*alpha_plus*v\
                    + 0.5*(body_force_np1 + body_force_n)\
                        *fe.dot( xi[5], fe.grad(v) ) )*fe.dx
    

    

# (Bi)linear forms for f6
a6 = alpha_plus**2*f6_trial*v*fe.dx\
    + alpha_plus*fe.dot( xi[6], fe.grad(v) ) * f6_trial * fe.dx\
        + alpha_plus*fe.dot( xi[6], fe.grad(f6_trial) )*v*fe.dx\
            + fe.dot( xi[6], fe.grad(f6_trial) ) * fe.dot( xi[6], fe.grad(v) )*fe.dx 

body_force_np1 = body_Force_extrap(f_list_n, f_list_n_1, 6, Force_density)
body_force_n = body_Force(vel(f_list_n), 6, Force_density)

L6 = ( alpha_minus*alpha_plus*f6_n*v\
    + alpha_minus*f6_n*fe.dot( xi[6], fe.grad(v) )\
    +   (1/tau)*( f_equil_extrap(f_list_n, f_list_n_1, 6) + f_equil(f_list_n, 6) ) * alpha_plus*v\
    + (1/tau)*( f_equil_extrap(f_list_n, f_list_n_1, 6) + f_equil(f_list_n, 6) ) * fe.dot( xi[6], fe.grad(v) )\
        - fe.dot( xi[6], fe.grad(f6_n) )*alpha_plus*v\
            - fe.dot( xi[6], fe.grad(f6_n) )*fe.dot( xi[6], fe.grad(v) )\
                + 0.5*(body_force_np1 + body_force_n)*alpha_plus*v\
                    + 0.5*(body_force_np1 + body_force_n)\
                        *fe.dot( xi[6], fe.grad(v) ) )*fe.dx
    
    
# (Bi)linear forms for f7
a7 = alpha_plus**2*f7_trial*v*fe.dx\
    + alpha_plus*fe.dot( xi[7], fe.grad(v) ) * f7_trial * fe.dx\
        + alpha_plus*fe.dot( xi[7], fe.grad(f7_trial) )*v*fe.dx\
            + fe.dot( xi[7], fe.grad(f7_trial) ) * fe.dot( xi[7], fe.grad(v) )*fe.dx 

body_force_np1 = body_Force_extrap(f_list_n, f_list_n_1, 7, Force_density)
body_force_n = body_Force(vel(f_list_n), 7, Force_density)

L7 = ( alpha_minus*alpha_plus*f7_n*v\
    + alpha_minus*f7_n*fe.dot( xi[7], fe.grad(v) )\
    +   (1/tau)*( f_equil_extrap(f_list_n, f_list_n_1, 7) + f_equil(f_list_n, 7) ) * alpha_plus*v\
    + (1/tau)*( f_equil_extrap(f_list_n, f_list_n_1, 7) + f_equil(f_list_n, 7) ) * fe.dot( xi[7], fe.grad(v) )\
        - fe.dot( xi[7], fe.grad(f7_n) )*alpha_plus*v\
            - fe.dot( xi[7], fe.grad(f7_n) )*fe.dot( xi[7], fe.grad(v) )\
                + 0.5*(body_force_np1 + body_force_n)*alpha_plus*v\
                    + 0.5*(body_force_np1 + body_force_n)\
                        *fe.dot( xi[7], fe.grad(v) ) )*fe.dx
    
    
# (Bi)linear forms for f8
a8 = alpha_plus**2*f8_trial*v*fe.dx\
    + alpha_plus*fe.dot( xi[8], fe.grad(v) ) * f8_trial * fe.dx\
        + alpha_plus*fe.dot( xi[8], fe.grad(f8_trial) )*v*fe.dx\
            + fe.dot( xi[8], fe.grad(f8_trial) ) * fe.dot( xi[8], fe.grad(v) )*fe.dx

body_force_np1 = body_Force_extrap(f_list_n, f_list_n_1, 8, Force_density)
body_force_n = body_Force(vel(f_list_n), 8, Force_density)

L8 = ( alpha_minus*alpha_plus*f8_n*v\
    + alpha_minus*f8_n*fe.dot( xi[8], fe.grad(v) )\
    +   (1/tau)*( f_equil_extrap(f_list_n, f_list_n_1, 8) + f_equil(f_list_n, 8) ) * alpha_plus*v\
    + (1/tau)*( f_equil_extrap(f_list_n, f_list_n_1, 8) + f_equil(f_list_n, 8) ) * fe.dot( xi[8], fe.grad(v) )\
        - fe.dot( xi[8], fe.grad(f8_n) )*alpha_plus*v\
            - fe.dot( xi[8], fe.grad(f8_n) )*fe.dot( xi[8], fe.grad(v) )\
                + 0.5*(body_force_np1 + body_force_n)*alpha_plus*v\
                    + 0.5*(body_force_np1 + body_force_n)\
                        *fe.dot( xi[8], fe.grad(v) ) )*fe.dx


# Assemble matrices
A0, A1, A2 = fe.assemble(a0), fe.assemble(a1), fe.assemble(a2)
A3, A4, A5 = fe.assemble(a3), fe.assemble(a4), fe.assemble(a5)
A6, A7, A8 = fe.assemble(a6), fe.assemble(a7), fe.assemble(a8)
    
    
for n in range(1, num_steps):
    # Update current time
    t += dt
    
    # Assemble right-hand side vectors
    b0, b1, b2 = fe.assemble(L0), fe.assemble(L1), fe.assemble(L2)
    b3, b4, b5 = fe.assemble(L3), fe.assemble(L4), fe.assemble(L5)
    b6, b7, b8 = fe.assemble(L6), fe.assemble(L7), fe.assemble(L8)
    
    # Apply BCs for distribution functions 5, 2, and 6
    # Apply BCs for distribution functions 5, 2, and 6
    L_bc_f0.apply(A0, b0)
    L_bc_f1.apply(A1, b1)
    L_bc_f2.apply(A2, b2)
    L_bc_f3.apply(A3, b3)
    L_bc_f4.apply(A4, b4)
    L_bc_f5.apply(A5, b5)
    L_bc_f6.apply(A6, b6)
    L_bc_f7.apply(A7, b7)
    L_bc_f8.apply(A8, b8)
    
    # Apply BCs for distribution functions 7, 4, 8
    U_bc_f0.apply(A0, b0)
    U_bc_f1.apply(A1, b1)
    U_bc_f2.apply(A2, b2)
    U_bc_f3.apply(A3, b3)
    U_bc_f4.apply(A4, b4)
    U_bc_f5.apply(A5, b5)
    U_bc_f6.apply(A6, b6)
    U_bc_f7.apply(A7, b7)
    U_bc_f8.apply(A8, b8)
    
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
    
    f0_n_1.assign(f0_n)
    f1_n_1.assign(f1_n)
    f2_n_1.assign(f2_n)
    f3_n_1.assign(f3_n)
    f4_n_1.assign(f4_n)
    f5_n_1.assign(f5_n)
    f6_n_1.assign(f6_n)
    f7_n_1.assign(f7_n)
    f8_n_1.assign(f8_n)
    
    f0_n.assign(f0)
    f1_n.assign(f1)
    f2_n.assign(f2)
    f3_n.assign(f3)
    f4_n.assign(f4)
    f5_n.assign(f5)
    f6_n.assign(f6)
    f7_n.assign(f7)
    f8_n.assign(f8)
    
    fe.project(w[0]*fe.Constant(rho_wall), V, function=f0_lower_func)
    fe.project(w[1]*fe.Constant(rho_wall), V, function=f1_lower_func)
    fe.project(w[2]*fe.Constant(rho_wall), V, function=f2_lower_func)
    fe.project(w[3]*fe.Constant(rho_wall), V, function=f3_lower_func)
    fe.project(w[4]*fe.Constant(rho_wall), V, function=f4_lower_func)
    fe.project(w[5]*fe.Constant(rho_wall), V, function=f5_lower_func)
    fe.project(w[6]*fe.Constant(rho_wall), V, function=f6_lower_func)
    fe.project(w[7]*fe.Constant(rho_wall), V, function=f7_lower_func)
    fe.project(w[8]*fe.Constant(rho_wall), V, function=f8_lower_func)
    
    fe.project(w[0]*fe.Constant(rho_wall), V, function=f0_upper_func)
    fe.project(w[1]*fe.Constant(rho_wall), V, function=f1_upper_func)
    fe.project(w[2]*fe.Constant(rho_wall), V, function=f2_upper_func)
    fe.project(w[3]*fe.Constant(rho_wall), V, function=f3_upper_func)
    fe.project(w[4]*fe.Constant(rho_wall), V, function=f4_upper_func)
    fe.project(w[5]*fe.Constant(rho_wall), V, function=f5_upper_func)
    fe.project(w[6]*fe.Constant(rho_wall), V, function=f6_upper_func)
    fe.project(w[7]*fe.Constant(rho_wall), V, function=f7_upper_func)
    fe.project(w[8]*fe.Constant(rho_wall), V, function=f8_upper_func)



u_expr = vel(f_list_n)
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
f_list = [f0, f1, f2, f3, f4, f5, f6, f7, f8]
f_grids = []
for idx, fi in enumerate(f_list):
    # flatten values, sort into (y,x) lex order, then reshape into (ny, nx)
    fi_vals   = fi.vector().get_local()
    fi_sorted = fi_vals[order_f]
    fi_grid   = fi_sorted.reshape((ny_f, nx_f))
    f_grids.append(fi_grid)
    # Optional: if you want to name them individually:
    # globals()[f"f{idx}_grid"] = fi_grid

# Now f_grids[i] is the (ny_f × nx_f) array of f_i values at the mesh grid.
# e.g., f_grids[0] is f0_grid, f_grids[1] is f1_grid, etc.

