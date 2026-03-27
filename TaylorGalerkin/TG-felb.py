import fenics as fe
import os
import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

start_time = time.time()

comm = fe.MPI.comm_world
rank = fe.MPI.rank(comm)

plt.close('all')


T = 100000
dt = 0.1

num_steps = int(np.ceil(T/dt))


Re = 0.96
L_x = 32
L_y = 32
nx = 5
ny = 5
h = L_x/nx

Force_density = np.array([2.6014e-5, 0.0])

# Where to save the plots
WORKDIR = os.getcwd()
outDirName = os.path.join(WORKDIR, "update")#f"Lx{L_x}_Ly{L_y}_nx{nx}_ny{ny}_dt{dt}_force{Force_density[0]}")
os.makedirs(outDirName, exist_ok=True)

# Lattice speed of sound
c_s = np.sqrt(1/3)
tau = 1

# Number of discrete velocities
Q = 9


# Density on wall
rho_wall = 1.0
# Initial density
rho_init = 1.0
u_wall = (0.0, 0.0)

u_max = Force_density[0]*L_y**2/(8 * rho_init * tau/3)


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

# Set up domain. For simplicity, do rectangular mesh.
mesh = fe.RectangleMesh(comm, fe.Point(0, 0), fe.Point(L_x, L_y), nx, nx)

# Set periodic boundary conditions at left and right endpoints
class PeriodicBoundaryX(fe.SubDomain):
    def inside(self, point, on_boundary):
        return fe.near(point[0], 0.0) and on_boundary

    def map(self, right_bdy, left_bdy):
        left_bdy[0] = right_bdy[0] - L_x
        left_bdy[1] = right_bdy[1]


pbc = PeriodicBoundaryX()


V = fe.FunctionSpace(mesh, "P", 1, constrained_domain=pbc)
Vvec = fe.VectorFunctionSpace(mesh, "P", 1, constrained_domain=pbc)
vel_n = fe.Function(Vvec)
vel_star = fe.Function(Vvec)


# Define trial and test functions, as well as
# finite element functions at previous timesteps
f_trial = fe.TrialFunction(V)
f_n = []
for idx in range(Q):
    f_n.append(fe.Function(V))
rho_n = fe.Function(V)
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
def getDens(f_list):
    return f_list[0] + f_list[1] + f_list[2] + f_list[3] + f_list[4]\
        + f_list[5] + f_list[6] + f_list[7] + f_list[8]

# Define velocity


def getVel(f_list):
    distr_fn_sum = f_list[0]*xi[0] + f_list[1]*xi[1] + f_list[2]*xi[2]\
        + f_list[3]*xi[3] + f_list[4]*xi[4] + f_list[5]*xi[5]\
        + f_list[6]*xi[6] + f_list[7]*xi[7] + f_list[8]*xi[8]

    density = getDens(f_list)

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
def f_equil(rho, vel, vel_idx):

    u   = vel   
    ci       = xi[vel_idx]
    cu = fe.dot(ci, u)
    u2 = fe.dot(u, u)

    feq = w[vel_idx] * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)

    return feq


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


# Define boundary conditions. Here we will use bounceback BCs

# For distributions 5, 2, and 6, the conjugate distributions 
# are 7, 4, and 8, respectively.
tol = 1e-8
def Bdy_Lower(x, on_boundary):
    if on_boundary:
        if fe.near(x[1], 0, tol):
            return True
        else:
            return False
    else:
        return False

f5_lower = f_n[7]  
f2_lower = f_n[4] 
f6_lower = f_n[8] 

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
# at the upper wall. Here, the conjugate distributions are 
# 5, 2, and 6. 
tol = 1e-8
def Bdy_Upper(x, on_boundary):
    if on_boundary:
        if fe.near(x[1], L_y, tol):
            return True
        else:
            return False
    else:
        return False

f7_upper = f_n[5]  
f4_upper = f_n[2]  
f8_upper = f_n[6]  

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

advection_forms = []
double_advection_forms = []

# Define linear and bilinear forms for the collision and streaming steps
for idx in range(Q):

    bilinear_forms_stream.append(f_trial * v * fe.dx)
    bilinear_forms_collision.append(f_trial*v*fe.dx)
    
    advection_forms.append( v*fe.dot(xi[idx], fe.grad(f_trial))*fe.dx )
    double_advection_forms.append( fe.dot(xi[idx],fe.grad(v))\
                                  *fe.dot(xi[idx],fe.grad(f_trial))*fe.dx )

    #double_dot_product_term = -0.5*dt**2 * fe.dot(xi[idx], fe.grad(f_star[idx]))\
    #    * fe.dot(xi[idx], fe.grad(v)) * fe.dx

    dot_product_force_term = 0.5*dt**2 * fe.dot(xi[idx], fe.grad(v))\
        * body_Force(vel_star, idx, Force_density) * fe.dx

    lin_form_stream = dt*v*body_Force(vel_star, idx, Force_density)*fe.dx\
        + dot_product_force_term

    lin_form_coll = (f_n[idx] - dt/tau * (f_n[idx] - f_equil(rho_n, vel_n, idx)) )*v*fe.dx

    linear_forms_stream.append(lin_form_stream)
    linear_forms_collision.append(lin_form_coll)

# Assemble matrices for first step
sysMatStream = []
sysMatColl = []
solverListStream = []
solverListColl = []
rhsVecStreaming = []
rhsVecCollision = []
advectionMats = []
doubleAdvectionMats = []
for idx in range(Q):
    sysMatStream.append(fe.assemble(bilinear_forms_stream[idx]))
    sysMatColl.append(fe.assemble(bilinear_forms_collision[idx]))
    rhsVecStreaming.append( fe.assemble(linear_forms_stream[idx]) )
    rhsVecCollision.append( fe.assemble(linear_forms_collision[idx]) )
    
    advectionMats.append(fe.assemble(advection_forms[idx]))
    doubleAdvectionMats.append(fe.assemble(double_advection_forms[idx]))

for idx in range(Q):
    A = sysMatStream[idx]
    solver = fe.LUSolver(A)
    solverListStream.append(solver)
    
    A2 = sysMatColl[idx]
    solver2 = fe.LUSolver(A2)
    solverListColl.append(solver2)


vel_file = fe.XDMFFile(comm, f"{outDirName}/vel.xdmf")
vel_file.parameters["flush_output"] = True
vel_file.parameters["functions_share_mesh"] = True
vel_file.parameters["rewrite_function_mesh"] = False


# Apply BCs to matrices for distribution functions 5, 2, and 6
bc_f5.apply(sysMatStream[5])
bc_f5.apply(advectionMats[5])
bc_f5.apply(doubleAdvectionMats[5])

bc_f2.apply(sysMatStream[2])
bc_f2.apply(advectionMats[2])
bc_f2.apply(doubleAdvectionMats[2])

bc_f6.apply(sysMatStream[6])
bc_f6.apply(advectionMats[6])
bc_f6.apply(doubleAdvectionMats[6])

# Apply BCs to matrices for distribution functions 7, 4, 8
bc_f7.apply(sysMatStream[7])
bc_f7.apply(advectionMats[7])
bc_f7.apply(doubleAdvectionMats[7])

bc_f4.apply(sysMatStream[4])
bc_f4.apply(advectionMats[4])
bc_f4.apply(doubleAdvectionMats[4])

bc_f8.apply(sysMatStream[8])
bc_f8.apply(advectionMats[8])
bc_f8.apply(doubleAdvectionMats[8])

prevTimeMat = fe.assemble(f_trial*v*fe.dx)
streamingPrevTimeVecs=  [0]*Q
advectionVecs = [0]*Q
doubleAdvectionVecs =[0]*Q
    

xi_arr = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],
                   [1,1],[-1,1],[-1,-1],[1,-1]], dtype=float)
    
# Timestepping
t = 0.0
for n in range(num_steps):
    t += dt
    
    pre_coll_time = time.time()
    # We will try to do collision locally, since it is a pure
    # time-dependnet ODE
    f_vals = np.array([f_n[idx].vector().get_local() for idx in range(Q)])
    xi_arr = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],
                   [1,1],[-1,1],[-1,-1],[1,-1]], dtype=float)

    # Compute rho and u as numpy arrays over all DOFs
    rho = f_vals.sum(axis=0)                          # shape (n_dofs,)
    ux  = (xi_arr[:,0,None] * f_vals).sum(axis=0) / rho + Force_density[0]*dt/(2*rho)
    uy  = (xi_arr[:,1,None] * f_vals).sum(axis=0) / rho + Force_density[1]*dt/(2*rho)
    vel = np.stack([ux, uy])
    cu = xi_arr[:,0,None]*ux + xi_arr[:,1,None]*uy        # (9, n_dofs)
    u2 = ux**2 + uy**2                                    # (n_dofs,)
    feq = w[:,None] * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)

    f_star_np = f_vals - (dt/tau)*(f_vals - feq)
    [f_star[idx].vector().set_local(f_star_np[idx,:]) for idx in range(Q)]
    vel_star.vector().set_local(np.stack([ux, uy], axis=1).flatten())
    post_coll_time = time.time()
    #print("collision_time =", post_coll_time - pre_coll_time)

    # Assemble RHS vectors for streaming step
    pre_stream_time = time.time()
    for idx in range(Q):
        streamingPrevTimeVecs[idx]=(prevTimeMat*f_star[idx].vector())
        advectionVecs[idx]=(advectionMats[idx]*f_star[idx].vector())
        doubleAdvectionVecs[idx]=(doubleAdvectionMats[idx]*f_star[idx].vector())
        
        fe.assemble(linear_forms_stream[idx], tensor=rhsVecStreaming[idx])
        rhsVecStreaming[idx]+= streamingPrevTimeVecs[idx]\
            -dt*advectionVecs[idx] -0.5*dt**2*doubleAdvectionVecs[idx]
    post_assemble_stream_time = time.time() 
    #print("stream assemble =", post_assemble_stream_time - pre_stream_time)


    pre_assign_time = time.time()
    f5_lower_func.assign(f_star[7] )
    f2_lower_func.assign(f_star[4] )
    f6_lower_func.assign(f_star[8] )
    f7_upper_func.assign(f_star[5] )
    f4_upper_func.assign(f_star[2] )
    f8_upper_func.assign(f_star[6] )
    post_assign_time = time.time()
    #print("assign time = ", post_assign_time - pre_assign_time)

    pre_apply_time = time.time()
    # Apply BCs for distribution functions 5, 2, and 6
    bc_f5.apply(rhsVecStreaming[5])
    bc_f2.apply(rhsVecStreaming[2])
    bc_f6.apply(rhsVecStreaming[6])

    # Apply BCs for distribution functions 7, 4, 8
    bc_f7.apply(rhsVecStreaming[7])
    bc_f4.apply(rhsVecStreaming[4])
    bc_f8.apply(rhsVecStreaming[8])
    post_apply_time = time.time()
    #print("time to apply BCs ", post_apply_time - pre_apply_time)

    pre_stream_time = time.time()
    # Solve linear system for streaming step
    for idx in range(Q):
        solverListStream[idx].solve(f_nP1[idx].vector(), rhsVecStreaming[idx])
    post_stream_time = time.time()
    #print("time to solve stream sys ", post_stream_time - pre_stream_time, "\n\n\n\n")


    # Update previous solutions

    for idx in range(Q):
        f_n[idx].assign(f_nP1[idx])
        
    a=1
        
    #fe.project(getVel(f_n), Vvec, function=vel_n)
    #fe.project(getDens(f_n), V, function=rho_n)

    if n % 5000 == 0:
        vel_expr = getVel(f_n)
        fe.project(vel_expr, Vvec, function=vel_n)
        vel_file.write(vel_n, t)
        
        # print("max |drho|   =", np.max(np.abs(rho_diff)), flush=True)
        # print("max |d_momentum_x|=", np.max(np.abs(momx_diff)), flush=True)
        # print("max |d_momentum_y|=", np.max(np.abs(momy_diff)), flush=True)

        # log_file.write(f"{np.max(np.abs(rho_diff)):15.4f} {np.max(np.abs(momx_diff)):15.4f}  {np.max(np.abs(momy_diff)):15.4f} \n")
        # log_file.flush()
        
        u_new, v_new = 0, 0
        
        for i in range(Q):
            xi_new = xi[i].values()
            u_new += f_n[i].vector().get_local()*xi_new[0]
            v_new += f_n[i].vector().get_local()*xi_new[1]

        u_e = fe.Expression('u_max*( 1 - pow( (2*x[1]/L_y -1), 2 ) )',
                            degree=2, u_max=u_max, L_y=L_y)
        u_e = fe.interpolate(u_e, V)
        error = np.linalg.norm(u_e.vector().get_local() - u_new)
        time_elapsed = time.time() - start_time
        print('t = %.4f: error = %.3g' % (t, error), flush=True)
        print('max u:', u_new.max(), flush=True)
        print("Time elapsed = ", time_elapsed, "\n\n", flush=True)

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
        for i in range(num_points_analytical):
            u_ex[i] = (1 - (2*y_values_analytical[i]/L_y - 1)**2)

        for point in points:
            u_expr = getVel(f_n)
            V_vec = fe.VectorFunctionSpace(mesh, "P", 1, constrained_domain=pbc)
            u = fe.project(u_expr, V_vec)
            u_at_point = u(point)
            u_x_values.append(u_at_point[0] / u_max)



        fig_name = "felb_dt" + str(dt) + "_simTime" + str(n) + ".png"
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

