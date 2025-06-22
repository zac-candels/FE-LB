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
    weights_k = weights[k]  # corresponding weight
    cs2 = 1.0 / 3.0
    prefactor = (1.0 - dt / (2.0 * tau)) * weights_k

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
    feq = weights[k] * rho_vals * (
        1 + 3*disc_vel_u + 4.5*disc_vel_u**2 - 1.5*uu
    )

    return feq