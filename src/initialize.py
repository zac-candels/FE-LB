import fenics as fe


def initializeDistributions(f_n, Force_density, V, latticeClass, c_s, dt):
    
    xi = latticeClass.xi 
    w = latticeClass.weights
    for idx in range(len(f_n)):
        f_n[idx] = (fe.project(f_equil_init(idx,
                                            Force_density,
                                            dt, 
                                            xi, 
                                            w, 
                                            c_s), V))
        
    return f_n

def initializeDistributionsMultiPhase(f_n, forceDensity, V, latticeClass,
                                      c_s, dt, tau):
    
    xi = latticeClass.xi 
    w = latticeClass.weights
    for idx in range(len(f_n)):
        f_n[idx] = (fe.project(f_equil_init_multiPhase(idx,
                                                       forceDensity,
                                                       dt,
                                                       xi,
                                                       w,
                                                       c_s,
                                                       tau), V))
        
    return f_n

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


def f_equil_init_multiPhase(vel_idx, forceDensity, dt, xi, w, c_s, tau):
    rho_init = fe.Constant(1.0)
    rho_expr = fe.Constant(1.0)
    c_s2 = c_s**c_s
    dt = fe.Constant(dt)

    vel_0 = - (dt/2)*forceDensity/rho_init
    
    vel_grad = fe.grad(vel_0)

    ci = xi[vel_idx]
    ci_dot_u = fe.dot(ci, vel_0)
    
    
    f_eq  = w[vel_idx] * rho_expr * (
        1
        + ci_dot_u / c_s**2
        + ci_dot_u**2 / (2*c_s**4)
        - fe.dot(vel_0, vel_0) / (2*c_s**2)
    )
    
    c_c_outer = fe.outer(ci, ci)
    
    I = fe.Identity(2)
    
    Q = c_c_outer - c_s2 * I
    
    F_u_outer1 = fe.outer( forceDensity, vel_0 )
    u_F_outer2 = fe.outer(vel_0, forceDensity) 
    force_vel_outer = F_u_outer1 + u_F_outer2
    c_dot_F = fe.inner( ci, forceDensity)
    
    f_neq = - w[vel_idx]*tau/c_s2 * rho_expr * fe.inner(Q, vel_grad)\
        - w[vel_idx]*dt/(2*c_s2) * ( c_dot_F\
                                 + 1/(2*c_s2) * fe.inner(Q, force_vel_outer) )
    
    
    return f_eq + f_neq