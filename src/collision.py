import fenics as fe 
import numpy as np 


def collideLocal(f_n, f_star, latticeClass, Force, tau, dt):
    
    xi_arr = latticeClass.xi_arr
    Q = latticeClass.Q
    w = latticeClass.weights
    c_s = latticeClass.c_s
    
    forceVals_x, forceVals_y = Force[0], Force[1]
    
    f_vals = np.array([f_n[idx].vector().get_local() for idx in range(Q)])
    
    rho = f_vals.sum(axis=0)                          # shape (n_dofs,)
    ux  = (xi_arr[:,0,None] * f_vals).sum(axis=0) / rho + forceVals_x*dt/(2*rho)
    uy  = (xi_arr[:,1,None] * f_vals).sum(axis=0) / rho + forceVals_y*dt/(2*rho)
    vel = np.stack([ux, uy])
    cu = xi_arr[:,0,None]*ux + xi_arr[:,1,None]*uy        # (9, n_dofs)
    u2 = ux**2 + uy**2                                    # (n_dofs,)
    feq = w[:,None] * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)
    
    # Now for the foce term
    u_dot_F = ux * forceVals_x + uy * forceVals_y   # (n_dofs,)
    ck_dot_F = xi_arr @ np.column_stack((forceVals_x,forceVals_y)).T   # shape (Q, n_dofs)
    
    ck_dot_u = xi_arr[:,0,None]*ux + xi_arr[:,1,None]*uy   # (9, n_dofs) -- same as your 'cu'

    force_term = w[:, None] * (
          ck_dot_F / c_s**2
        + (ck_dot_u * ck_dot_F) / c_s**4   # ← ck_dot_u, not u_dot_F
        - u_dot_F[None, :] / c_s**2        # ← minus sign
    )
    

    f_star_np = f_vals - dt/tau*(f_vals - feq) + dt*force_term
    [f_star[idx].vector().set_local(f_star_np[idx,:]) for idx in range(Q)]
    
    return f_star
        
        