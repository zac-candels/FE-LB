import fenics as fe
import os
import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.close('all')

# Where to save the plots
WORKDIR = os.getcwd()
outDirName = os.path.join(WORKDIR, "figures")
os.makedirs(outDirName, exist_ok=True)

T = 1800

Re = 0.96
nx = ny = np.array([4, 8, 12, 16])
err = np.zeros(len(nx))
conv_rate = np.zeros(len(nx))
L_x = 32
L_y = 32
h = L_x/nx



f_conv = open("conv_rate.txt", "w")
f_conv.write("nx   h   err_infty  conv_rate \n")
for i in range(len(nx)):

    dt = 0.001 * h[i]
    num_steps = int(np.ceil(T/dt))

    # Lattice speed of sound
    c_s = np.sqrt(1/3)  # np.sqrt( 1./3. * h**2/dt**2 )

    nu = 1.0/3.0
    tau = nu/c_s**2 + 0.5*dt

    # Number of discrete velocities
    Q = 9
    Force_density = np.array([2.6041666e-5, 0.0])

    # Density on wall
    rho_wall = 1.0
    # Initial density
    rho_init = 1.0
    u_wall = (0.0, 0.0)

    #nu = tau/3.
    u_max = Force_density[0]*L_y**2/(8*rho_init*nu)


    # D2Q9 lattice velocities
    xi = [
        fe.Constant((0.0,  0.0)),
        fe.Constant((1.0,  0.0)),
        fe.Constant((0.0,  1.0)),
        fe.Constant((-1.0,  0.0)),
        fe.Constant((0.0, -1.0)),
        fe.Constant((1.0,  1.0)),
        fe.Constant((-1.0,  1.0)),
        fe.Constant((-1.0, -1.0)),
        fe.Constant((1.0, -1.0)),
    ]

    # Corresponding weights
    w = np.array([
        4/9,
        1/9, 1/9, 1/9, 1/9,
        1/36, 1/36, 1/36, 1/36
    ])

    # Set up domain. For simplicity, do unit square mesh.

    mesh = fe.RectangleMesh(fe.Point(0, 0), fe.Point(L_x, L_y), nx[i], ny[i])

    # Set periodic boundary conditions at left and right endpoints


    class PeriodicBoundaryX(fe.SubDomain):
        def inside(self, point, on_boundary):
            return fe.near(point[0], 0.0) and on_boundary

        def map(self, right_bdy, left_bdy):
            # Map left boundary to the right
            left_bdy[0] = right_bdy[0] - L_x
            left_bdy[1] = right_bdy[1]


    pbc = PeriodicBoundaryX()


    V = fe.FunctionSpace(mesh, "P", 1, constrained_domain=pbc)


    # Define trial and test functions, as well as
    # finite element functions at previous timesteps

    f_trial = fe.TrialFunction(V)
    f_n = []
    for idx in range(Q):
        f_n.append(fe.Function(V))

    v = fe.TestFunction(V)

    # Define FE functions to hold post-streaming solution at nP1 timesteps
    f_nP1 = []
    for idx in range(Q):
        f_nP1.append(fe.Function(V))

    # Define FE functions to hold post-collision distributions
    f_star = []
    for idx in range(Q):
        f_star.append(fe.Function(V))


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

        F = fe.Constant((Force_density[0], Force_density[1]))
        vel_term2 = F * dt / (2 * density)

        return vel_term1 + vel_term2


    # Define initial equilibrium distributions
    def f_equil_init(vel_idx, Force_density):
        rho_init = fe.Constant(1.0)
        rho_expr = fe.Constant(1.0)

        vel_0 = -fe.Constant((Force_density[0]*dt/(2*rho_init),
                            Force_density[1]*dt/(2*rho_init)))

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


    # def f_equil(f_list, vel_idx):
    #     rho_expr = sum(fj for fj in f_list)
    #     u_expr = vel(f_list)
    #     ci = xi[vel_idx]
    #     ci_dot_u = fe.dot(ci, u_expr)
    #     return w[vel_idx] * rho_expr * (
    #         1
    #         + ci_dot_u / c_s**2
    #         + ci_dot_u**2 / (2*c_s**4)
    #         - fe.dot(u_expr, u_expr) / (2*c_s**2)
    #     )

    xi_array = np.array([[float(c.values()[0]), float(c.values()[1])] for c in xi])

    def f_equil(f_list, idx):
        """
        Compute equilibrium distribution for direction idx
        Returns a NumPy array (values at all DoFs).
        """
        # Number of DoFs
        N = f_list[0].vector().size()

        # Stack all f_i values: shape (Q, N)
        f_stack = np.array([f.vector().get_local() for f in f_list])

        # Compute density at each DoF
        rho_vec = np.sum(f_stack, axis=0)  # shape (N,)

        # Compute velocity at each DoF
        ux_vec = np.sum(f_stack * xi_array[:,0][:,None], axis=0)
        uy_vec = np.sum(f_stack * xi_array[:,1][:,None], axis=0)

        u2 = ux_vec**2 + uy_vec**2

        # Compute ci . u for this direction
        cu = xi_array[idx,0]*ux_vec + xi_array[idx,1]*uy_vec

        feq = w[idx] * rho_vec * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)

        return feq  # NumPy array


    # Define collision operator


    def coll_op(f_list, vel_idx):
        return -(f_list[vel_idx] - f_equil(f_list, vel_idx)) / (tau + 0.5)


    def body_Force(vel, vel_idx, Force_density):
        prefactor = w[vel_idx]
        inverse_cs2 = 1 / c_s**2
        inverse_cs4 = 1 / c_s**4

        xi_dot_prod_F = xi[vel_idx][0]*Force_density[0]\
            + xi[vel_idx][1]*Force_density[1]

        u_dot_prod_F = vel[0]*Force_density[0] + vel[1]*Force_density[1]

        xi_dot_u = xi[vel_idx][0]*vel[0] + xi[vel_idx][1]*vel[1]

        Force = prefactor*(inverse_cs2*(xi_dot_prod_F - u_dot_prod_F)
                        + inverse_cs4*xi_dot_u*xi_dot_prod_F)

        return Force

    # # Initialize distribution functions. We will use
    # f_i^{0} \gets f_i^{0, eq}( \rho_0, \bar{u}_0 ),
    # where \bar{u}_0 = u_0 - F\Delta t/( 2 \rho_0 ).
    # Here we will take u_0 = 0.


    for idx in range(Q):
        f_n[idx] = (fe.project(f_equil_init(idx, Force_density), V))


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


    rho_expr = sum(fk for fk in f_n)

    f5_lower = f_n[7]  # rho_expr
    f2_lower = f_n[4]  # rho_expr
    f6_lower = f_n[8]  # rho_expr

    f5_lower_func = fe.Function(V)
    f2_lower_func = fe.Function(V)
    f6_lower_func = fe.Function(V)

    fe.project(f5_lower, V, function=f5_lower_func)
    fe.project(f2_lower, V, function=f2_lower_func)
    fe.project(f6_lower, V, function=f6_lower_func)

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


    rho_expr = sum(fk for fk in f_n)

    f7_upper = f_n[5]  # rho_expr
    f4_upper = f_n[2]  # rho_expr
    f8_upper = f_n[6]  # rho_expr

    f7_upper_func = fe.Function(V)
    f4_upper_func = fe.Function(V)
    f8_upper_func = fe.Function(V)

    fe.project(f7_upper, V, function=f7_upper_func)
    fe.project(f4_upper, V, function=f4_upper_func)
    fe.project(f8_upper, V, function=f8_upper_func)

    bc_f7 = fe.DirichletBC(V, f7_upper_func, Bdy_Upper)
    bc_f4 = fe.DirichletBC(V, f4_upper_func, Bdy_Upper)
    bc_f8 = fe.DirichletBC(V, f8_upper_func, Bdy_Upper)

    # Define variational problems

    bilinear_forms_stream = []
    linear_forms_stream = []

    bilinear_forms_collision = []
    linear_forms_collision = []

    n = fe.FacetNormal(mesh)
    opp_idx = {0: 0, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}

    for idx in range(Q):

        bilinear_forms_stream.append(f_trial * v * fe.dx)

        double_dot_product_term = -0.5*dt**2 * fe.dot(xi[idx], fe.grad(f_star[idx]))\
            * fe.dot(xi[idx], fe.grad(v)) * fe.dx

        dot_product_force_term = 0.5*dt**2 * fe.dot(xi[idx], fe.grad(v))\
            * body_Force(vel(f_star), idx, Force_density) * fe.dx

        if idx in opp_idx:
            # UFL scalar: dot product with facet normal
            dot_xi_n = fe.dot(xi[idx], n)

            # indicator = 1.0 when dot_xi_n < 0 (incoming), else 0.0
            indicator = fe.conditional(fe.lt(dot_xi_n, 0.0),
                                    fe.Constant(1.0),
                                    fe.Constant(0.0))

            # build surface term only for incoming distributions
            surface_term = 0.5*dt**2 * v * fe.dot(xi[idx], fe.grad(f_n[opp_idx[idx]])) \
                * dot_xi_n * indicator * fe.ds
        else:
            # no surface contribution for this idx
            surface_term = fe.Constant(0.0) * v * fe.ds

        lin_form_idx = f_star[idx]*v*fe.dx\
            - dt*v*fe.dot(xi[idx], fe.grad(f_star[idx]))*fe.dx\
            + dt*v*body_Force(vel(f_star), idx, Force_density)*fe.dx\
            + double_dot_product_term\
            + dot_product_force_term + surface_term

        linear_forms_stream.append(lin_form_idx)

    # Assemble matrices for first step
    sys_mat = []
    rhs_vec_streaming = [0]*Q
    rhs_vec_collision = [0]*Q
    for idx in range(Q):
        sys_mat.append(fe.assemble(bilinear_forms_stream[idx]))

    solver_list = []
    for idx in range(Q):
        A = sys_mat[idx]

        # Create CG solver
        solver = fe.KrylovSolver("cg", "ilu")  # use ILU preconditioner
        solver.set_operator(A)

        # Optional: set solver parameters
        prm = solver.parameters
        prm["absolute_tolerance"] = 1e-12
        prm["relative_tolerance"] = 1e-8
        prm["maximum_iterations"] = 1000
        prm["nonzero_initial_guess"] = False

        solver_list.append(solver)

    # Timestepping
    t = 0.0
    for n in range(num_steps):

        t += dt

        for idx in range(Q):
            f_eq_vec = f_equil(f_n, idx)
            #f_eq_vec = f_eq.vector().get_local()
            f_n_vec = f_n[idx].vector().get_local()
            
            f_new = f_n_vec - dt/tau * (f_n_vec - f_eq_vec)
        
            f_star[idx].vector().set_local(f_new)
            f_star[idx].vector().apply("insert")

        # Assemble RHS vectors
        for idx in range(Q):
            rhs_vec_streaming[idx] = (fe.assemble(linear_forms_stream[idx]))

        f5_lower_func.vector()[:] = f_star[7].vector()[:]
        f2_lower_func.vector()[:] = f_star[4].vector()[:]
        f6_lower_func.vector()[:] = f_star[8].vector()[:]
        f7_upper_func.vector()[:] = f_star[5].vector()[:]
        f4_upper_func.vector()[:] = f_star[2].vector()[:]
        f8_upper_func.vector()[:] = f_star[6].vector()[:]

        # Apply BCs for distribution functions 5, 2, and 6
        bc_f5.apply(sys_mat[5], rhs_vec_streaming[5])
        bc_f2.apply(sys_mat[2], rhs_vec_streaming[2])
        bc_f6.apply(sys_mat[6], rhs_vec_streaming[6])

        # Apply BCs for distribution functions 7, 4, 8
        bc_f7.apply(sys_mat[7], rhs_vec_streaming[7])
        bc_f4.apply(sys_mat[4], rhs_vec_streaming[4])
        bc_f8.apply(sys_mat[8], rhs_vec_streaming[8])

        # Solve linear system in each timestep
        for idx in range(Q):
            solver_list[idx].solve(f_nP1[idx].vector(), rhs_vec_streaming[idx])


        # Update previous solutions

        for idx in range(Q):
            f_n[idx].assign(f_nP1[idx])
            
        error = 0
        if n % num_steps - 1 == 0:
            u_new, v_new = 0, 0
            
            for ii in range(Q):
                xi_new = xi[ii].values()
                u_new += f_n[ii].vector().get_local()*xi_new[0]
                v_new += f_n[ii].vector().get_local()*xi_new[1]

            u_e = fe.Expression('u_max*( 1 - pow( (2*x[1]/L_y -1), 2 ) )',
                                degree=2, u_max=u_max, L_y=L_y)
            u_e = fe.interpolate(u_e, V)
            error = np.linalg.norm(u_e.vector().get_local() - u_new)
            print('t = %.4f: error = %.3g' % (t, error))
            print('max u:', np.max(u_new))
            err[i] = error
            
            if i > 0:
                conv_rate[i] = np.log10(err[i]/err[i-1])/np.log10(h[i]/h[i-1])
                
                
            file_arg = f"{nx[i]:10d} {h[i]:15.6e} {err[i]:15.6e} {conv_rate[i]:15.6f}\n"
            f_conv.write(file_arg)
    
    u_expr = vel(f_n)
    V_vec = fe.VectorFunctionSpace(mesh, "P", 2, constrained_domain=pbc)
    u = fe.project(u_expr, V_vec)
    num_points_analytical = 200
    num_points_numerical = 10
    y_values_analytical = np.linspace(0, L_y, num_points_analytical)
    y_values_numerical = np.linspace(0, L_y, num_points_numerical)
    x_fixed = L_x/2
    points = [(x_fixed, y) for y in y_values_numerical]
    u_x_values = []
    u_ex = np.linspace(0, L_y, num_points_analytical)
    nu = tau/3
    u_max = Force_density[0]*L_y**2/(8*rho_init*nu)
    for j in range(num_points_analytical):
        u_ex[j] = (1 - (2*y_values_analytical[j]/L_y - 1)**2)

    for point in points:
        u_at_point = u(point)
        u_x_values.append(u_at_point[0] / u_max)


    WORKDIR = os.getcwd()  # this will be correct if you `cd` into /root/shared
    outDirName = os.path.join(WORKDIR, "figures")
    os.makedirs(outDirName, exist_ok=True)
    fig_name = "nx = " + str(nx[i]) + ".png"
    output = os.path.join(outDirName, fig_name)

    plt.figure()
    plt.plot(y_values_numerical/L_y, u_x_values, 'o', label="FE soln.")
    plt.plot(y_values_analytical/L_y, u_ex, label="Analytical soln.")
    plt.ylabel(r"$u_x/u_{{max}}$", fontsize=20)
    plt.xlabel(r"$y/L_y$", fontsize=20)
    plt.legend()
    plt.tick_params(direction="in")


    print("Saving figure to:", os.path.abspath(output))
    plt.savefig(output, dpi=400, format='png', bbox_inches='tight')
        
f_conv.close()

