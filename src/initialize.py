import fenics as fe

def f_equil_init(vel_idx, Force_density, dt, xi, w, c_s):
    rho_init = fe.Constant(1.0)
    rho_expr = fe.Constant(1.0)

    vel_0 = -fe.Constant((Force_density.values()[0]*dt/(2*rho_init),
                          Force_density.values()[1]*dt/(2*rho_init)))

    # u_expr = fe.project(V_vec, vel_0)

    ci = xi[vel_idx]
    ci_dot_u = fe.dot(ci, vel_0)
    return w[vel_idx] * rho_expr * (
        1
        + ci_dot_u / c_s**2
        + ci_dot_u**2 / (2*c_s**4)
        - fe.dot(vel_0, vel_0) / (2*c_s**2)
    )