import fenics as fe
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

T = 2000
dt = 1
num_steps = int(np.ceil(T/dt))


Re = 0.96
nx = ny = np.array([5, 10, 15, 20])
err = np.zeros(4)
conv_rate = np.zeros(4)
L_x = L_y = 32
h = L_x/nx
for i in range(len(nx)):
    
    # Lattice speed of sound
    c_s = np.sqrt(1/3) # np.sqrt( 1./3. * h**2/dt**2 )
    
    nu = 1.0/6.0
    #tau = nu/c_s**2 + dt/2 
    tau = 1
    u_max = 0.01
    
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
    
    mesh = fe.RectangleMesh(fe.Point(0,0), fe.Point(L_x, L_y), nx[i], nx[i] )
    
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
    
    
    
    
    # Define trial and test functions, as well as 
    # finite element functions at previous timesteps
    
    f_trial = []
    f_n = []
    for idx in range(Q):
        f_trial.append(fe.TrialFunction(V))
        f_n.append(fe.Function(V))
        
    v = fe.TestFunction(V)
    
    
    
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
        
        
    # # Initialize distribution functions. We will use 
    # f_i^{0} \gets f_i^{0, eq}( \rho_0, \bar{u}_0 ),
    # where \bar{u}_0 = u_0 - F\Delta t/( 2 \rho_0 ).
    # Here we will take u_0 = 0.
    
    for idx in range(Q):
        f_n[idx] = (  fe.project(f_equil_init(idx, Force_density), V )  )
    
    
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
        
    rho_expr = sum( fk for fk in f_n )
     
    f5_lower = f_n[7] # rho_expr
    f2_lower = f_n[4] # rho_expr 
    f6_lower = f_n[8] # rho_expr
    
    f5_lower_func = fe.Function(V)
    f2_lower_func = fe.Function(V)
    f6_lower_func = fe.Function(V)
    
    fe.project( f5_lower, V, function=f5_lower_func )
    fe.project( f2_lower, V, function=f2_lower_func )
    fe.project( f6_lower, V, function=f6_lower_func )
    
    bc_f5 = fe.DirichletBC(V, f5_lower_func, Bdy_Lower)
    bc_f2 = fe.DirichletBC(V, f2_lower_func, Bdy_Lower)
    bc_f6 = fe.DirichletBC(V, f6_lower_func, Bdy_Lower)
    
    # Similarly, we will define boundary conditions for f_7, f_4, and f_8
    # at the upper wall. Once again, boundary conditions simply reduce
    # to \rho * w_i
    
    
    tol = 1e-8
    def Bdy_Upper(x, on_boundary):
        if on_boundary:
            if fe.near(x[1], L_y, tol):
                return True
            else:
                return False
        else:
            return False
    
    rho_expr = sum( fk for fk in f_n )
     
    f7_upper = f_n[5] # rho_expr
    f4_upper = f_n[2] # rho_expr 
    f8_upper = f_n[6] # rho_expr
    
    f7_upper_func = fe.Function(V)
    f4_upper_func = fe.Function(V)
    f8_upper_func = fe.Function(V)
    
    fe.project( f7_upper, V, function=f7_upper_func )
    fe.project( f4_upper, V, function=f4_upper_func )
    fe.project( f8_upper, V, function=f8_upper_func )
    
    bc_f7 = fe.DirichletBC(V, f7_upper_func, Bdy_Upper)
    bc_f4 = fe.DirichletBC(V, f4_upper_func, Bdy_Upper)
    bc_f8 = fe.DirichletBC(V, f8_upper_func, Bdy_Upper)
    
    # Define variational problems
    
    bilinear_forms_step1 = []
    linear_forms_step1 = []
    
    for idx in range(Q):
        bilinear_forms_step1.append( f_trial[idx] * v * fe.dx\
                                         + dt*fe.dot( xi[idx], fe.grad(f_trial[idx]) )\
                                             * v * fe.dx )
        linear_forms_step1.append( ( f_n[idx] + dt*coll_op(f_n, idx)\
          + dt * body_Force( vel(f_n), idx, Force_density) ) * v * fe.dx )
            
    # Assemble matrices for first step
    sys_mat_step1 = []
    rhs_vec_step1 = [0]*Q
    for idx in range(Q):
        sys_mat_step1.append( fe.assemble( bilinear_forms_step1[idx] ) )
    
    # Define FE functions to hold solution at nP1 timesteps
    f_nP1 = []
    for idx in range(Q):
        f_nP1.append(fe.Function(V))
    
        
    t = 0
    for n in range(1):
        t += dt
        
        for idx in range(Q):
            rhs_vec_step1[idx] = ( fe.assemble( linear_forms_step1[idx] ) )
            
        # Apply BCs for distribution functions 5, 2, and 6
        bc_f5.apply(sys_mat_step1[5], rhs_vec_step1[5])
        bc_f2.apply(sys_mat_step1[2], rhs_vec_step1[2])
        bc_f6.apply(sys_mat_step1[6], rhs_vec_step1[6])
        
        # Apply BCs for distributions 7, 4, 8
        bc_f7.apply(sys_mat_step1[7], rhs_vec_step1[7])
        bc_f4.apply(sys_mat_step1[4], rhs_vec_step1[4])
        bc_f8.apply(sys_mat_step1[8], rhs_vec_step1[8])
        
        for idx in range(Q):
            fe.solve( sys_mat_step1[idx], f_nP1[idx].vector(), rhs_vec_step1[idx] )
            
        fe.project(f_n[7], V, function=f5_lower_func)
        fe.project(f_n[4], V, function=f2_lower_func)
        fe.project(f_n[8], V, function=f6_lower_func)
        fe.project(f_n[5], V, function=f7_upper_func)
        fe.project(f_n[2], V, function=f4_upper_func)
        fe.project(f_n[6], V, function=f8_upper_func)
            
    # Now define finite element functions for the n - 1 timestep
    
    f_nM1 = []
    for idx in range(Q):
        f_nM1.append( fe.Function(V) )
        
    # Assign initial values to f_nM1
    for idx in range(Q):
        f_nM1[idx].assign( f_n[idx] )
        
    # Update f_n
    for idx in range(Q):
        f_n[idx].assign( f_nP1[idx] )   
        
    
    bilinear_forms_step2 = []
    linear_forms_step2 = []
    
    # Define variational problems for step 2 (CN timestep)
    
    for idx in range(Q):
        bilinear_forms_step2.append( alpha_plus**2*f_trial[idx]*v*fe.dx\
            + alpha_plus*fe.dot( xi[idx], fe.grad(v) ) * f_trial[idx] * fe.dx\
                + alpha_plus*fe.dot( xi[idx], fe.grad(f_trial[idx]) )*v*fe.dx\
                    + fe.dot( xi[idx], fe.grad(f_trial[idx]) )\
                        *fe.dot( xi[idx], fe.grad(v) )*fe.dx )
    
        body_force_np1 = body_Force_extrap(f_n, f_nM1, idx, Force_density)
        body_force_n = body_Force(vel(f_n), idx, Force_density)
    
        linear_forms_step2.append( ( alpha_minus*alpha_plus*f_n[idx]*v\
            + alpha_minus*f_n[idx]*fe.dot( xi[idx], fe.grad(v) )\
            +   (1/tau)*( f_equil_extrap(f_n, f_nM1, idx) + f_equil(f_n, idx) ) * alpha_plus*v\
            + (1/tau)*( f_equil_extrap(f_n, f_nM1, idx) + f_equil(f_n, idx) ) * fe.dot( xi[idx], fe.grad(v) )\
                - fe.dot( xi[idx], fe.grad(f_n[idx]) )*alpha_plus*v\
                    - fe.dot( xi[idx], fe.grad(f_n[idx]) )*fe.dot( xi[idx], fe.grad(v) )\
                        + 0.5*(body_force_np1 + body_force_n)*alpha_plus*v\
                            + 0.5*(body_force_np1 + body_force_n)\
                                *fe.dot( xi[idx], fe.grad(v) ) )*fe.dx )
    
    
    # Assemble matrices for CN timestep
    sys_mat_step2 = []
    rhs_vec_step2 = [0]*Q
    for idx in range(Q):
        sys_mat_step2.append( fe.assemble(bilinear_forms_step2[idx] ) )
        
        
    # CN timestepping
    for n in range(1, num_steps):
        
        # Assemble RHS vectors
        for idx in range(Q):
            rhs_vec_step2[idx] = ( fe.assemble(linear_forms_step2[idx]) )
            
        # Apply BCs for distribution functions 5, 2, and 6
        bc_f5.apply(sys_mat_step2[5], rhs_vec_step2[5])
        bc_f2.apply(sys_mat_step2[2], rhs_vec_step2[2])
        bc_f6.apply(sys_mat_step2[6], rhs_vec_step2[6])
        
        # Apply BCs for distribution functions 7, 4, 8
        bc_f7.apply(sys_mat_step2[7], rhs_vec_step2[7])
        bc_f4.apply(sys_mat_step2[4], rhs_vec_step2[4])
        bc_f8.apply(sys_mat_step2[8], rhs_vec_step2[8])
        
        # Solve linear system in each timestep
        for idx in range(Q):
            fe.solve( sys_mat_step2[idx], f_nP1[idx].vector(), rhs_vec_step2[idx] )
            
        # Update previous solutions
        for idx in range(Q):
            f_nM1[idx].assign( f_n[idx] )
        
        for idx in range(Q):
            f_n[idx].assign( f_nP1[idx] )
            
        fe.project(f_n[7], V, function=f5_lower_func)
        fe.project(f_n[4], V, function=f2_lower_func)
        fe.project(f_n[8], V, function=f6_lower_func)
        fe.project(f_n[5], V, function=f7_upper_func)
        fe.project(f_n[2], V, function=f4_upper_func)
        fe.project(f_n[6], V, function=f8_upper_func)
        
        u_expr = vel(f_n)
        V_vec = fe.VectorFunctionSpace(mesh, "P", 2, constrained_domain=pbc)
        u_n = fe.project(u_expr, V_vec)
        u_n_x = fe.project(u_n.split()[0], V)
        
        u_e = fe.Expression('u_max*( 1 - pow( (2*x[1]/L_x -1), 2 ) )',
                                     degree = 2, u_max = u_max, L_x = L_x)
        u_e = fe.interpolate(u_e, V)
        error = np.abs(u_e.vector().get_local() - u_n_x.vector().get_local()).max()
        #print('t = %.4f: error = %.3g' % (t, error))
        #print('max u:', u_n_x.vector().get_local().max())
    err[i] = error
    if i > 0:
        conv_rate[i] = np.log10(err[i]/err[i-1])/np.log10(h[i]/h[i-1])
        

    

    
