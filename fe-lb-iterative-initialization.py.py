import fenics as fe
import numpy as np
import matplotlib.pyplot as plt

# Simulating Poiseuille flow by solving the lattice Boltzmann 
# equation using the finite element method, using D2Q9.

# Our velocity set is the following:
# c_1 = (1, 0), w_1 = 1/9
# c_2 = (0, 1), w_2 = 1/9
# c_3 = (-1, 0), w_3 = 1/9
# c_4 = (0, -1), w_4 = 1/9
# c_5 = (1, 1), w_5 = 1/36
# c_6 = (-1, 1), w_6 = 1/36
# c_7 = (-1, -1), w_7 = 1/36
# c_8 = (1, -1), w_8 = 1/36
# c_9 = (0, 0), w_9 = 4/9


def compute_collision(fk_vec, fk_eq_vec, tau=1.0):
    J_vec = fk_vec.copy()
    J_vec.axpy(-1.0, fk_eq_vec)     # J_vec = fk_vec - fk_eq_vec
    J_vec *= -1.0 / tau             # J_vec = -(1/τ)(fk - fk_eq)
    return J_vec

def compute_force(u, F, k, dt, tau):
    """
    Compute S_k at the DoFs for the k-th velocity direction in D2Q9.

    Args:
        u: Function (vector-valued, dim=2) for macroscopic velocity.
        F: Either a constant NumPy array [Fx, Fy] or FEniCS Function/Expression.
        k: Index of the D2Q9 direction (0 ≤ k ≤ 8).
        dt: Time step
        tau: Relaxation time

    Returns:
        NumPy array with values of S_k at each DoF.
    """
    V = u.function_space().sub(0).collapse()
    mesh = V.mesh()
    dof_coords = V.tabulate_dof_coordinates().reshape((-1, 2))

    u_vals = np.array([u(x) for x in dof_coords])  # shape: (n_dof, 2)

    # Evaluate forcing field F at each point
    if isinstance(F, fe.Function) or isinstance(F, fe.Expression):
        F_vals = np.array([F(x) for x in dof_coords])
    else:
        F_vals = np.tile(np.asarray(F), (len(dof_coords), 1))  # broadcast constant force

    disc_vel_k = disc_vel[k]  # D2Q9 velocity vector for direction k
    w_k = w[k]  # corresponding weight
    cs2 = 1.0 / 3.0
    prefactor = (1.0 - dt / (2.0 * tau)) * w_k

    # Compute dot products for each term
    S_vals = []

    for i in range(len(dof_coords)):
        u_i = u_vals[i]
        F_i = F_vals[i]

        c_dot_F = np.dot(disc_vel_k, F_i)
        c_dot_u = np.dot(disc_vel_k, u_i)
        u_dot_F = np.dot(u_i, F_i)
        cuF_term = (np.dot(disc_vel_k, F_i)) / cs2
        second_term = ((np.outer(disc_vel_k, disc_vel_k) - cs2 * np.identity(2)) @ u_i) @ F_i / cs2**2

        S_ki = prefactor * (cuF_term + second_term)
        S_vals.append(S_ki)

    return np.array(S_vals)


def compute_feq(rho, u_expr, k):
    """
    Compute f_k^{eq} over the DoFs for the k-th velocity direction.
    
    Args:
        rho: Function (FEniCS) for density ρ(x)
        u_expr: Expression or Function for velocity u(x)
        k: int in 0..8 for the D2Q9 direction index

    Returns:
        NumPy array with values of f_k^{eq} at each DoF.
    """
    V = rho.function_space()
    mesh = V.mesh()
    dof_coords = V.tabulate_dof_coordinates().reshape((-1, 2))

    # Evaluate rho at DoFs
    rho_vals = rho.vector().get_local()

    # Evaluate velocity at DoFs
    if isinstance(u_expr, fe.Expression):
        u_vals = np.array([u_expr(x) for x in dof_coords])
    else:  # assume u_expr is a Function
        u_vals = np.array([u_expr(x) for x in dof_coords])

    # Extract lattice velocity vector c_k
    disc_vel_k = disc_vel[k]

    # Compute dot products
    disc_vel_u = np.dot(u_vals, disc_vel_k)
    uu = np.sum(u_vals**2, axis=1)

    # Compute f_eq using standard LBM formula
    feq = w[k] * rho_vals * (
        1 + 3*disc_vel_u + 4.5*disc_vel_u**2 - 1.5*uu
    )

    return feq

    
# D2Q9 lattice velocities
disc_vel = np.array([
    [0,0], [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1]
])

# Corresponding weights
w = np.array([
    4/9,
    1/9, 1/9, 1/9, 1/9,
    1/36, 1/36, 1/36, 1/36
])


T = 10.0 
num_steps = 1000
dt = T / num_steps
tau = 1.0
Q = 9
F = np.array([3.0, 0.0])

nx = ny = 32
mesh = fe.RectangleMesh(fe.Point(0,0), fe.Point(2.0, 0.4), nx, ny)
V = fe.FunctionSpace(mesh, "P", 1)
V_vec = fe.VectorFunctionSpace(mesh, "P", 1)

#bc_expression1 = "w_i*rho*( 1 + (u_x*c_disc_vel + u_y*c_yi)/(c_s*c_s)   )"
#bc_expression2 = " + w_i*rho*( pow( u_x*c_disc_vel + u_y*c_yi , 2 )/(2*pow(c_s,4)) )"
#bc_expression3 = " + w_i*rho( - (u_x*u_x + u_y*u_y)/(2*c_s*c_s)  )"

f0_D = fe.Expression( "(4/9)*rho", degree = 2, rho = 1.0)
f1_D = fe.Expression( "(1/9)*rho", degree = 2, rho = 1.0)
f2_D = fe.Expression( "(1/9)*rho", degree = 2, rho = 1.0)
f3_D = fe.Expression( "(1/9)*rho", degree = 2, rho = 1.0)
f4_D = fe.Expression( "(1/9)*rho", degree = 2, rho = 1.0)
f5_D = fe.Expression( "(1/36)*rho", degree = 2, rho = 1.0)
f6_D = fe.Expression( "(1/36)*rho", degree = 2, rho = 1.0)
f7_D = fe.Expression( "(1/36)*rho", degree = 2, rho = 1.0)
f8_D = fe.Expression( "(1/36)*rho", degree = 2, rho = 1.0)


def boundary(x, on_boundary):
    return on_boundary

bc_f0 = fe.DirichletBC(V, f0_D, boundary)
bc_f1 = fe.DirichletBC(V, f1_D, boundary)
bc_f2 = fe.DirichletBC(V, f2_D, boundary)
bc_f3 = fe.DirichletBC(V, f3_D, boundary)
bc_f4 = fe.DirichletBC(V, f4_D, boundary)
bc_f5 = fe.DirichletBC(V, f5_D, boundary)
bc_f6 = fe.DirichletBC(V, f6_D, boundary)
bc_f7 = fe.DirichletBC(V, f7_D, boundary)
bc_f8 = fe.DirichletBC(V, f8_D, boundary)


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
f = [f0_n, f1_n, f2_n, f3_n, f4_n, f5_n, f6_n, f7_n, f8_n]

distr_fn = fe.TrialFunction(V) 
v = fe.TestFunction(V)

a = distr_fn*v*fe.dx 
#b = [c[i][0]*fe.grad(distr_fn)[0]*v*fe.dx for i in range(Q)]
#c = [c[i][1]*fe.grad(distr_fn)[1]*v*fe.dx for i in range(Q)]
#j = [(-1/tau)*(f[i] - f_eq[i]) for i in range(Q)]

A_mat = fe.assemble(a)
B_mats = []
C_mats = []
J_vecs = []

for k in range(Q):
    if( disc_vel[k][0] != 0):
        B_mat = fe.assemble( disc_vel[k][0]*fe.grad(distr_fn)[0]*v*fe.dx )
    else:
        B_mat = fe.assemble( distr_fn*v*fe.dx)
        B_mat *= 0
       
    if( disc_vel[k][1] != 0):
        C_mat = fe.assemble( disc_vel[k][1]*fe.grad(distr_fn)[1]*v*fe.dx )
    else:
        C_mat = fe.assemble( distr_fn*v*fe.dx )
        C_mat *= 0
        
    J_vec = fe.assemble( (-1/tau)*(f[k] - f_eq[k])*fe.dx )
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
        rho.vector().axpy(1.0, f[k].vector())
        
    # Step 2: Update f_eq using current rho and fixed u0
    for k in range(Q):
        f_eq[k].vector().set_local( compute_feq(rho, u0, k) )
        
    delta_f = 0.0 
    
    # Step 3: update f_k
    for k in range(Q):
        J_vec = compute_collision(f[k].vector(), f_eq[k].vector(), tau)
        S_vec = compute_force(u0_fn, F, k, dt, tau)
        
        rhs = fe.Vector(f[k].vector())
        rhs *= 1.0 
        
        rhs.axpy(-dt, B_mats[k]*f[k].vector())
        rhs.axpy(-dt, C_mats[k]*f[k].vector())
        rhs.axpy(dt, J_vec)
        
        f_new = fe.Vector(f[k].vector())
        fe.solve(A_mat, f_new, rhs)
        
        delta_f = np.linalg.norm(f_new - f[k].vector() )#max(delta_f, np.linalg.norm(f_new - f[k].vector(), ord=np.Inf))
        
        f[k].vector()[:] = f_new
        
    print("df = ", delta_f)
    counter += 1
        
        
# Now that the distribution functions have been initialized, we can 
# proceed to time-stepping.
   
    
    
    
    
    
        
        
    
    


