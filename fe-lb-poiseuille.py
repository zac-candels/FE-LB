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
    xi_k = xi[k]

    # Compute dot products
    xi_u = np.dot(u_vals, xi_k)
    uu = np.sum(u_vals**2, axis=1)

    # Compute f_eq using standard LBM formula
    feq = w[k] * rho_vals * (
        1 + 3*xi_u + 4.5*xi_u**2 - 1.5*uu
    )

    return feq

    
# D2Q9 lattice velocities
xi = np.array([
    [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1], [0,0]
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

nx = ny = 32
mesh = fe.RectangleMesh(fe.Point(0,0), fe.Point(2.0, 0.4), nx, ny)
V = fe.FunctionSpace(mesh, "P", 1)

#bc_expression1 = "w_i*rho*( 1 + (u_x*c_xi + u_y*c_yi)/(c_s*c_s)   )"
#bc_expression2 = " + w_i*rho*( pow( u_x*c_xi + u_y*c_yi , 2 )/(2*pow(c_s,4)) )"
#bc_expression3 = " + w_i*rho( - (u_x*u_x + u_y*u_y)/(2*c_s*c_s)  )"

f1_D = fe.Expression( "(1/9)*rho", degree = 2, rho = 1.0)
f2_D = fe.Expression( "(1/9)*rho", degree = 2, rho = 1.0)
f3_D = fe.Expression( "(1/9)*rho", degree = 2, rho = 1.0)
f4_D = fe.Expression( "(1/9)*rho", degree = 2, rho = 1.0)
f5_D = fe.Expression( "(1/36)*rho", degree = 2, rho = 1.0)
f6_D = fe.Expression( "(1/36)*rho", degree = 2, rho = 1.0)
f7_D = fe.Expression( "(1/36)*rho", degree = 2, rho = 1.0)
f8_D = fe.Expression( "(1/36)*rho", degree = 2, rho = 1.0)
f9_D = fe.Expression( "(4/9)*rho", degree = 2, rho = 1.0)

def boundary(x, on_boundary):
    return on_boundary

bc_f1 = fe.DirichletBC(V, f1_D, boundary)
bc_f2 = fe.DirichletBC(V, f2_D, boundary)
bc_f3 = fe.DirichletBC(V, f3_D, boundary)
bc_f4 = fe.DirichletBC(V, f4_D, boundary)
bc_f5 = fe.DirichletBC(V, f5_D, boundary)
bc_f6 = fe.DirichletBC(V, f6_D, boundary)
bc_f7 = fe.DirichletBC(V, f7_D, boundary)
bc_f8 = fe.DirichletBC(V, f8_D, boundary)
bc_f9 = fe.DirichletBC(V, f9_D, boundary)

# Don't do anything for initial conditions because we're going
# to start off with a velocity field which is identically 0
# initially.

f1_n = fe.interpolate(f1_D, V)
f2_n = fe.interpolate(f2_D, V)
f3_n = fe.interpolate(f3_D, V)
f4_n = fe.interpolate(f4_D, V)
f5_n = fe.interpolate(f5_D, V)
f6_n = fe.interpolate(f6_D, V)
f7_n = fe.interpolate(f7_D, V)
f8_n = fe.interpolate(f8_D, V)
f9_n = fe.interpolate(f9_D, V)

f_eq = [f1_n, f2_n, f3_n, f4_n, f5_n, f6_n, f7_n, f8_n, f9_n]
f = [f1_n, f2_n, f3_n, f4_n, f5_n, f6_n, f7_n, f8_n, f9_n]

gamma = fe.TrialFunction(V) 
v = fe.TestFunction(V)

a = gamma*v*fe.dx 
#b = [c[i][0]*fe.grad(gamma)[0]*v*fe.dx for i in range(Q)]
#c = [c[i][1]*fe.grad(gamma)[1]*v*fe.dx for i in range(Q)]
#j = [(-1/tau)*(f[i] - f_eq[i]) for i in range(Q)]

A_mat = fe.assemble(a)
B_mats = []
C_mats = []
J_vecs = []

for k in range(Q):
    if( xi[k][0] != 0):
        B_mat = fe.assemble( xi[k][0]*fe.grad(gamma)[0]*v*fe.dx )
    else:
        B_mat = fe.assemble( gamma*v*fe.dx)
        B_mat *= 0
       
    if( xi[k][1] != 0):
        C_mat = fe.assemble( xi[k][1]*fe.grad(gamma)[1]*v*fe.dx )
    else:
        C_mat = fe.assemble( gamma*v*fe.dx )
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
        
        
    
   
    
    
    
    
    
        
        
    
    


