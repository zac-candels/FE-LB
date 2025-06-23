import numpy as np
import fenics as fe

# D2Q9 lattice velocities
disc_vel = np.array([
    [0,0], [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1]
])

# Corresponding weights
weights = np.array([
    4/9,
    1/9, 1/9, 1/9, 1/9,
    1/36, 1/36, 1/36, 1/36
])

def compute_velocity(f_list, rho, disc_vel, F, dt):
    """
    Compute macroscopic velocity u from the distribution functions f_k and body force F.

    Parameters:
        f_list : list of Function
            List of distribution functions f_k.
        rho : Function
            Macroscopic density field.
        disc_vel : np.ndarray
            Array of shape (9, 2) with discrete D2Q9 velocity vectors.
        F : Function
            Body force field (vector-valued).
        dt : float
            Time step size.

    Returns:
        u : Function
            Velocity field as a vector Function.
    """
    V = rho.function_space()
    mesh = V.mesh()
    V_vec = fe.VectorFunctionSpace(mesh, "P", 1)

    u = fe.Function(V_vec)
    u_vals = np.zeros((V.dim(), 2))  # 2D velocity components at each DoF

    # Extract data
    f_vals = [f_k.vector().get_local() for f_k in f_list]
    rho_vals = rho.vector().get_local()
    F_vals = F.vector().get_local().reshape((-1, 2))

    # Compute momentum sum: sum_k f_k * c_k
    for k in range(9):
        u_vals[:, 0] += f_vals[k] * disc_vel[k, 0]
        u_vals[:, 1] += f_vals[k] * disc_vel[k, 1]

    # Add force term: (dt / 2) * F
    u_vals += 0.5 * dt * F_vals

    # Divide by density, with safe handling
    rho_safe = np.where(rho_vals > 1e-12, rho_vals, 1.0)
    u_vals[:, 0] /= rho_safe
    u_vals[:, 1] /= rho_safe

    # Assign to FEniCS vector function
    u.vector().set_local(u_vals.flatten())
    return u


def compute_collision(fk_vec, fk_eq_vec, tau=1.0):
    J_vec = fk_vec.copy()
    J_vec.axpy(-1.0, fk_eq_vec)     # J_vec = fk_vec - fk_eq_vec
    J_vec *= -1.0 / tau             # J_vec = -(1/τ)(fk - fk_eq)
    return J_vec


def compute_force(u, F, k, dt, tau, V_f):
    """
    Compute the force source term S_k as a FEniCS vector compatible with the LBM finite element solver.

    Args:
        u: Function (vector-valued, dim=2) for macroscopic velocity.
        F: Either a constant NumPy array [Fx, Fy] or FEniCS Function/Expression.
        k: Index of the D2Q9 direction (0 ≤ k ≤ 8).
        dt: Time step
        tau: Relaxation time
        V_f: Function space for the f_k distribution function (e.g., same as f_list_n[k].function_space())

    Returns:
        A PETSc vector (dolfin.Vector) with values of S_k projected onto V_f.
    """
    mesh = V_f.mesh()

    # Discrete velocity and weight for direction k
    c_k = disc_vel[k]
    w_k = weights[k]
    cs2 = 1.0 / 3.0
    prefactor = (1.0 - dt / (2.0 * tau)) * w_k

    # Define a FEniCS Expression to represent S_k(x)
    if isinstance(F, fe.Function) or isinstance(F, fe.Expression):
        F_expr = F
    else:
        F_expr = fe.Constant(F)

    # Define the full symbolic expression of the force term at each point
    x = fe.SpatialCoordinate(mesh)
    u_expr = fe.interpolate(u, u.function_space())

    c_dot_F = c_k[0] * F_expr[0] + c_k[1] * F_expr[1]
    c_dot_u = c_k[0] * u_expr[0] + c_k[1] * u_expr[1]
    u_dot_F = u_expr[0] * F_expr[0] + u_expr[1] * F_expr[1]

    # Build tensor expression for (c_k ⊗ c_k - cs2 * I)
    c_tensor = fe.as_matrix([ [fe.Constant(c_k[i]*c_k[j]) for j in range(2) ]\
                             for i in range(2)])
    identity = fe.Identity(2)
    second_term_tensor = (c_tensor - cs2 * identity)

    second_term = fe.dot(second_term_tensor * u_expr, F_expr) / cs2**2
    S_expr = prefactor * (c_dot_F / cs2 + second_term)

    # Project the scalar source expression to the same function space as f_k
    S_proj = fe.project(S_expr, V_f)

    return S_proj.vector()


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
    feq = weights[k] * rho_vals * (
        1 + 3*disc_vel_u + 4.5*disc_vel_u**2 - 1.5*uu
    )

    return feq