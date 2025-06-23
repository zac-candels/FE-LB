import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import fem_lbm_lib as fl


# Simulating Poiseuille flow by solving the lattice Boltzmann 
# equation using the finite element method, using D2Q9.



T = 10.0 
num_steps = 100
dt = T / num_steps
tau = 1.0
Q = 9
F_np = np.array([3.0, 0.0])
alpha = ( 2/dt + 1/tau )
rho_wall = 1.0
u_wall = (0.0, 0.0)

nx = ny = 32
L_x, L_y = 2.0, 0.4
mesh = fe.RectangleMesh(fe.Point(0,0), fe.Point(L_x, L_y), nx, ny)

class PeriodicBoundaryX(fe.SubDomain):
    def inside(self, x, on_boundary):
        # Return True for "target" side (usually the left)
        return fe.near(x[0], 0.0) and on_boundary

    def map(self, x, y):
        # Map left boundary to the right
        y[0] = x[0] - L_x
        y[1] = x[1]

pbc = PeriodicBoundaryX()

V = fe.FunctionSpace(mesh, "P", 1, constrained_domain=pbc)
V_vec = fe.VectorFunctionSpace(mesh, "P", 1, constrained_domain=pbc)


#bc_expression1 = "w_i*rho*( 1 + (u_x*c_disc_vel + u_y*c_yi)/(c_s*c_s)   )"
#bc_expression2 = " + w_i*rho*( pow( u_x*c_disc_vel + u_y*c_yi , 2 )/(2*pow(c_s,4)) )"
#bc_expression3 = " + w_i*rho( - (u_x*u_x + u_y*u_y)/(2*c_s*c_s)  )"


# We will first initialize the distribution functions to their
# equilibrium values and then use an iterative procedure to 
# get the right density.

f0_D = fe.Expression( "(4./9.)*dens", degree = 2, dens = 1.0)
f1_D = fe.Expression( "(1./9.)*dens", degree = 2, dens = 1.0)
f2_D = fe.Expression( "(1./9.)*dens", degree = 2, dens = 1.0)
f3_D = fe.Expression( "(1./9.)*dens", degree = 2, dens = 1.0)
f4_D = fe.Expression( "(1./9.)*dens", degree = 2, dens = 1.0)
f5_D = fe.Expression( "(1./36.)*dens", degree = 2, dens = 1.0)
f6_D = fe.Expression( "(1./36.)*dens", degree = 2, dens = 1.0)
f7_D = fe.Expression( "(1./36.)*dens", degree = 2, dens = 1.0)
f8_D = fe.Expression( "(1./36.)*dens", degree = 2, dens = 1.0)


def boundary(x, on_boundary):
    return on_boundary

class UpperWall(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and fe.near(x[1], 0.4)

class LowerWall(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and fe.near(x[1], 0.0)

upper_wall = UpperWall()
lower_wall = LowerWall()

# fl.disc_vel is list of discrete velocities (cx, cy)
# For upper wall (y = 0.4), incoming directions have cy < 0
# For lower wall (y = 0), incoming directions have cy > 0

incoming_upper = [k for k, c in enumerate(fl.disc_vel) if c[1] < 0]
incoming_lower = [k for k, c in enumerate(fl.disc_vel) if c[1] > 0]


# Don't do anything for initial conditions because we're going
# to start off with a velocity field which is identically 0
# initially.

f0_n = fe.interpolate(f0_D, V)
f1_n = fe.interpolate(f1_D, V)
f2_n = fe.interpolate(f2_D, V)
f3_n = fe.interpolate(f3_D, V)
f4_n = fe.interpolate(f4_D, V)
f5_n = fe.interpolate(f5_D, V)
f6_n = fe.interpolate(f6_D, V)
f7_n = fe.interpolate(f7_D, V)
f8_n = fe.interpolate(f8_D, V)


f_eq = [f0_n, f1_n, f2_n, f3_n, f4_n, f5_n, f6_n, f7_n, f8_n]
f_list_n = [f0_n, f1_n, f2_n, f3_n, f4_n, f5_n, f6_n, f7_n, f8_n]
f_list_np1 = [fe.Function(V), fe.Function(V), fe.Function(V), fe.Function(V),\
              fe.Function(V), fe.Function(V), fe.Function(V),
              fe.Function(V), fe.Function(V)]


f_fn = fe.TrialFunction(V) 
v = fe.TestFunction(V)

a = f_fn*v*fe.dx 

A_mat = fe.assemble(a)
B_mats = []
C_mats = []
J_vecs = []

# Get local matrix (copy, as dense)
dense_array = A_mat.array()

# Compute condition number
condition_number = np.linalg.cond(dense_array)

for k in range(Q):
    if( fl.disc_vel[k][0] != 0):
        B_mat = fe.assemble( fl.disc_vel[k][0]*fe.grad(f_fn)[0]*v*fe.dx )
    else:
        B_mat = fe.assemble( f_fn*v*fe.dx)
        B_mat *= 0
       
    if( fl.disc_vel[k][1] != 0):
        C_mat = fe.assemble( fl.disc_vel[k][1]*fe.grad(f_fn)[1]*v*fe.dx )
    else:
        C_mat = fe.assemble( f_fn*v*fe.dx )
        C_mat *= 0
        
    J_vec = fe.assemble( (-1/tau)*(f_list_n[k] - f_eq[k])*fe.dx )
    B_mats.append(B_mat)
    C_mats.append(C_mat)
    J_vecs.append(J_vec)


# Initialization

counter = 0
delta_f = 100.0
tol = 1e-5 

rho = fe.Function(V)
u0 = fe.Expression( ("u0_x", "u0_y"), degree = 1, u0_x=0.0, u0_y=0.0)
u0_fn = fe.project(u0, V_vec)

dofmap = V.dofmap()
dofs = dofmap.dofs()
dof_coords = V.tabulate_dof_coordinates()

f_dofs = []
while abs(delta_f) > tol:
    
    # Step 1: compute rho(x) = \sum_k f_k(x)
    rho.vector().zero()
    for k in range(Q):
        rho.vector().axpy(1.0, f_list_n[k].vector())
    density = rho.vector().vec().getArray()
        
    # Step 2: Update f_eq using current rho and fixed u0
    for k in range(Q):
        f_eq[k].vector().set_local( fl.compute_feq(rho, u0, k) )
        A = f_eq[k].vector().vec().getArray()
        
    delta_f = 0.0 
    
    # Step 3: update f_k
    for k in range(Q):
        J_vec = fl.compute_collision(f_list_n[k].vector(), f_eq[k].vector(), tau)
        
        S_vec = fl.compute_force(u0_fn, F_np, k, dt, tau)
        
        rhs = fe.Vector(f_list_n[k].vector())
        rhs *= 1.0 
        
        rhs.axpy(-dt, B_mats[k]*f_list_n[k].vector())
        rhs.axpy(-dt, C_mats[k]*f_list_n[k].vector())
        rhs.axpy(dt, J_vec)
        
        rhs_vec = rhs.get_local()
        
        
        f_list_new = fe.Vector(f_list_n[k].vector())
        fe.solve(A_mat, f_list_new, rhs)
        
        arr = np.array(f_list_new.get_local())
        
        delta_f = max(delta_f, np.linalg.norm(f_list_new - f_list_n[k].vector(), ord=np.inf))
        
        f_list_n[k].vector()[:] = f_list_new
        
    print("df = ", delta_f)
    counter += 1
        
        
# Now that the distribution functions have been initialized, we can 
# proceed to time-stepping.

u = fe.Function(V)
rhs = [fe.Function(V).vector() for _ in range(Q)]
    
for n in range( int(T) ):
    # First, compute rho
    rho.vector().zero()
    
    for k in range(Q):
        
        rho.vector().axpy(1.0, f_list_n[k].vector())
        
    # Then, compute u
    
    u.vector().zero()
    
    F_fn = fe.Function(V_vec)
    F_fn.vector().set_local(F_np.flatten())
    F_fn.vector().apply("insert")
    
    u = fl.compute_velocity(f_list_n, rho, fl.disc_vel, F_fn, dt)
    
    u_fn = fe.project(u, V_vec)
    
    # Update f_eq
    for k in range(Q):
        f_eq[k].vector().set_local( fl.compute_feq(rho, u, k) )
        
    f_eq_wall = [fe.Constant(fl.weights[k]*rho_wall) for k in range(Q)]
    
    
    # Create right-hand side vector
        
    for k in range(Q):
        J_vec = fl.compute_collision(f_list_n[k].vector(), f_eq[k].vector(), tau)
        S_vec = fl.compute_force(u0_fn, F_np, k, dt, tau)
        
        rhs[k].zero()
        rhs[k].axpy(1.0, f_list_n[k].vector())
        
        rhs[k].axpy(-dt, B_mats[k]*f_list_n[k].vector())
        rhs[k].axpy(-dt, C_mats[k]*f_list_n[k].vector())
        rhs[k].axpy(dt, J_vec)
        
        
    # Apply boundary conditions and solve 
    # For this simplest case, I'm simply going to use 
    # equilibrium boundary conditions.
    
    bcs_k = []

    for k in incoming_upper:
        bc_upper = fe.DirichletBC(V, f_eq_wall[k], upper_wall)
        bcs_k.append(bc_upper)
    
    for k in incoming_lower:
        bc_lower = fe.DirichletBC(V, f_eq_wall[k], lower_wall)
        bcs_k.append(bc_lower)
        
    
    for k in range(Q):
        for bc in bcs_k:
            bc.apply(A_mat, rhs[k])
            fe.solve(A_mat, f_list_np1[k].vector(), rhs[k])
            f_list_n[k].assign(f_list_np1[k])
            
        
# u = fl.compute_velocity(f_list_n, rho, fl.disc_vel, F_fn, dt)   
# u_fn = fe.project(u, V_vec)

# import matplotlib.pyplot as plt
# import dolfin as fe

# # Define parameters based on your mesh
# Lx = mesh.coordinates()[:, 0].max()
# Ly = mesh.coordinates()[:, 1].max()
# x_mid = Lx / 2.0  # Midpoint along x

# # Sample points along vertical line at x = Lx/2
# n_points = 100
# y_vals = np.linspace(0, Ly, n_points)
# u_x_vals = []

# for y in y_vals:
#     point = fe.Point(x_mid, y)
#     try:
#         u_vec = u_fn(point)
#         u_x_vals.append(u_vec[0])
#     except RuntimeError:
#         # May occur if point is outside domain due to rounding
#         u_x_vals.append(np.nan)

# # Plot the profile
# plt.figure(figsize=(6, 4))
# plt.plot(u_x_vals, y_vals, label="Computed $u_x(y)$")
# plt.xlabel("Horizontal velocity $u_x$")
# plt.ylabel("Vertical coordinate $y$")
# plt.title("Poiseuille Flow Velocity Profile at $x = L_x/2$")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
    
    
    
    
    
        
        
    
    


