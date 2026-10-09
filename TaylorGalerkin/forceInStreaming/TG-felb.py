import fenics as fe
import os
import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import shutil

start_time = time.time()

comm = fe.MPI.comm_world
rank = fe.MPI.rank(comm)

plt.close('all')


T = 10000000

nu = float(1/3)
c_s = np.sqrt(1/3)
tau = 1



Re = 0.96
L_x = 32
L_y = 32
nx = 30
ny = 30
h = L_x/nx
dt = 0.00125*h

num_steps = int(np.ceil(T/dt))

Force_density = fe.Constant((2.601466e-5, 0.0))

# Where to save the plots
WORKDIR = os.getcwd()
outDirName = os.path.join(WORKDIR, f"tau{tau}_dt{dt:1g}_nx{nx}")
if os.path.exists(outDirName):
    shutil.rmtree(outDirName)
os.makedirs(outDirName, exist_ok=True)

# Lattice speed of sound



max_U_vels = []

# Number of discrete velocities
Q = 9


# Density on wall
rho_wall = 1.0
# Initial density
rho_init = 1.0
u_wall = (0.0, 0.0)

u_max = Force_density.values()[0]*L_y**2/(8 * rho_init * tau/3)


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

# Define FE functions for force-dependent quantities
forceFn0 = fe.Function(V)
forceFn1 = fe.Function(V)
forceVelFn = fe.Function(V)
forceDensity_x = fe.Function(V)
forceDensity_y = fe.Function(V)

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

    F = fe.Constant((Force_density.values()[0], Force_density.values()[1]))
    vel_term2 = F * dt / (2 * density)

    return vel_term1 + vel_term2


# Define initial equilibrium distributions
def f_equil_init(vel_idx, Force_density):
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

massForm = f_trial*v*fe.dx
massMat = fe.assemble(massForm)
mass_action_form = fe.action(massForm, fe.Constant(1))
M_lumped = fe.assemble(massForm)
M_lumped.zero()
M_lumped.set_diagonal(fe.assemble(mass_action_form))
M_vect = fe.assemble(mass_action_form)
M_petsc = fe.as_backend_type(M_vect).vec()

sys_mat = []
sys_mat2 = []
sysMatLumped = []
advectionMats = []
advectionTransposeMats = []
doubleAdvectionMats = []
for idx in range(Q):
    sysMatLumped.append(M_petsc.copy())
    advectionMats.append(fe.assemble(advection_forms[idx]))
    A_T = advectionMats[idx].copy()
    advectionTransposeMats.append(A_T)
    doubleAdvectionMats.append(fe.assemble(double_advection_forms[idx]))
massMat = fe.assemble(f_trial*v*fe.dx)

streamingPrevTimeVecs= [f_star[0].vector().copy() for _ in range(Q)]
advectionVecs = [f_star[0].vector().copy() for _ in range(Q)]
doubleAdvectionVecs =[f_star[0].vector().copy() for _ in range(Q)]
rhsVecStreaming = [f_star[0].vector().copy() for _ in range(Q)]
prevTimeAcVec = f_star[0].vector().copy()
rhsVecTemp = f_star[0].vector().copy()

forceVec_x = f_star[0].vector().copy()
forceVec_y = f_star[0].vector().copy()


vel_file = fe.XDMFFile(comm, f"{outDirName}/vel.xdmf")
vel_file.parameters["flush_output"] = True
vel_file.parameters["functions_share_mesh"] = True
vel_file.parameters["rewrite_function_mesh"] = False



forceVecs = [0,0]
forceVelVec = 0
doubleForceVelVecs = [0,0]

    

xi_arr = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],
                   [1,1],[-1,1],[-1,-1],[1,-1]], dtype=float)

inverse_cs2 = 1 / c_s**2
inverse_cs4 = 1 / c_s**4
    
# Timestepping
t = 0.0
tMax = 15000
n=0
while t < tMax:
    n+=1
    t += dt
    
    f_vals = np.array([f_n[idx].vector().get_local() for idx in range(Q)])
    
    fe.assemble(Force_density.values()[0]*v*fe.dx, tensor=forceVec_x )
    fe.assemble(v*fe.dx, tensor=forceVec_y)
    forceVec_y.vec().scale(0)
    
    fe.solve(massMat, forceDensity_x.vector(), forceVec_x)

    fe.solve(massMat, forceDensity_y.vector(), forceVec_y)
    
    forceVals_x = forceDensity_x.vector().get_local()
    #forceVals_x = forceVals_x.reshape((-1, mesh.geometry().dim()))
    
    forceVals_y = forceDensity_y.vector().get_local()
    #forceVals_y = forceVals_y.reshape((-1, mesh.geometry().dim()))

    # Compute rho and u as numpy arrays over all DOFs
    rho = f_vals.sum(axis=0)                          # shape (n_dofs,)
    ux  = (xi_arr[:,0,None] * f_vals).sum(axis=0) / rho + forceVals_x*dt/(2*rho)
    uy  = (xi_arr[:,1,None] * f_vals).sum(axis=0) / rho + forceVals_y*dt/(2*rho)
    # ux[wall_dofs] = 0.0
    # uy[wall_dofs] = 0.0
    vel = np.stack([ux, uy])
    cu = xi_arr[:,0,None]*ux + xi_arr[:,1,None]*uy        # (9, n_dofs)
    u2 = ux**2 + uy**2                                    # (n_dofs,)
    feq = w[:,None] * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)
    

    f_star_np = f_vals - dt/(tau)*(f_vals - feq)
    [f_star[idx].vector().set_local(f_star_np[idx,:]) for idx in range(Q)]
    rho = f_star_np.sum(axis=0)
    ux  = (xi_arr[:,0,None] * f_star_np).sum(axis=0) / rho + forceVals_x*dt/(2*rho)
    uy  = (xi_arr[:,1,None] * f_star_np).sum(axis=0) / rho + forceVals_y*dt/(2*rho)
    vel_star.vector().set_local(np.stack([ux, uy], axis=1).flatten())

    post_coll_time_lb = time.time()
    #print("collision_time =", post_coll_time_lb - pre_coll_time_lb)

    u_dot_prod_F = fe.dot(vel_star, Force_density)
    
    stream_FE_start_time = time.time()
    for idx in range(Q):
        
        if idx == 0:
            Force = -w[idx]*(inverse_cs2* u_dot_prod_F)

        else:
    
            xi_dot_prod_F = fe.dot( xi[idx], Force_density)
    
            xi_dot_u = fe.dot(xi[idx], vel_star)
    
            Force = w[idx]*(inverse_cs2*(xi_dot_prod_F - u_dot_prod_F)
                           + inverse_cs4*xi_dot_u*xi_dot_prod_F)
        
        advectionForceTerm = fe.assemble(
            fe.dot(xi[idx], fe.grad(v))* Force * fe.dx)
            
        basicForceTerm = fe.assemble(v*Force*fe.dx)
        
        M_lumped.mult(f_star[idx].vector(), streamingPrevTimeVecs[idx])
        advectionMats[idx].mult(f_star[idx].vector(), advectionVecs[idx])
        doubleAdvectionMats[idx].mult(f_star[idx].vector(), doubleAdvectionVecs[idx])

        rhsVecStreaming[idx].zero()
        rhsVecStreaming[idx].axpy(1.0, streamingPrevTimeVecs[idx])
        rhsVecStreaming[idx].axpy(-dt, advectionVecs[idx])
        rhsVecStreaming[idx].axpy(-0.5*dt**2, doubleAdvectionVecs[idx])
        
        rhsVecStreaming[idx].axpy(dt, basicForceTerm)
        rhsVecStreaming[idx].axpy(0.5*dt**2, advectionForceTerm)
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
        vi = fe.as_backend_type(rhsVecStreaming[idx]).vec()
        f_nP1[idx].vector().vec().pointwiseDivide(vi, sysMatLumped[idx])
        # solverListStream[idx].solve(f_nP1[idx].vector(), rhsVecStreaming[idx])
    post_stream_time = time.time()
    #print("time to solve stream sys ", post_stream_time - pre_stream_time, "\n\n\n\n")


    # Update previous solutions


        
    bc_f5.apply(f_nP1[5].vector())
    bc_f2.apply(f_nP1[2].vector())
    bc_f6.apply(f_nP1[6].vector())

    # Apply BCs for distribution functions 7, 4, 8
    bc_f7.apply(f_nP1[7].vector())
    bc_f4.apply(f_nP1[4].vector())
    bc_f8.apply(f_nP1[8].vector())
    
    for idx in range(Q):
        f_n[idx].assign(f_nP1[idx])
        
    a=1
        
    #fe.project(getVel(f_n), Vvec, function=vel_n)
    #fe.project(getDens(f_n), V, function=rho_n)
    if n%10000 == 0:

        u_new, v_new = 0, 0
        
        for i in range(Q):
            xi_new = xi[i].values()
            u_new += f_n[i].vector().get_local()*xi_new[0]
            v_new += f_n[i].vector().get_local()*xi_new[1]

        vel_n.vector().set_local(np.stack([u_new, v_new], axis=1).flatten())
        u_e = fe.Expression('u_max*( 1 - pow( (2*x[1]/L_y -1), 2 ) )',
                            degree=2, u_max=u_max, L_y=L_y)
        u_e = fe.interpolate(u_e, V)

        vel_file.write(vel_n, t)
        print("Interpolated error expression", flush=True)

        if rank == 0:
            error = np.linalg.norm(u_e.vector().get_local() - u_new)
            print('t = %.4f: error = %.3g' % (t, error), flush=True)
            print('max(u_num)/max(u_ex):', u_new.max()/u_max, flush=True)

    
        num_points_analytical = 200
        num_points_numerical = 21
        y_values_analytical = np.linspace(0, L_y, num_points_analytical)
        y_values_numerical = np.linspace(0, L_y, num_points_numerical)
        x_fixed = L_x/2
        points = [(x_fixed, y) for y in y_values_numerical]
        u_x_values = []
        u_ex = np.linspace(0, L_y, num_points_analytical)

        for i in range(num_points_analytical):
            u_ex[i] = (1 - (2*y_values_analytical[i]/L_y - 1)**2)
    
        for point in points:
            u_at_point = vel_n(point)
            u_x_values.append(u_at_point[0] / u_max)

    
        if rank == 0:
            fig_name = "felb_dt" + str(dt) + "_simTime" + str(n) + ".png"
            output = os.path.join(outDirName, fig_name)
    
            plt.figure()
            plt.plot(y_values_numerical/L_y, u_x_values, 'o', label="FE soln.")
            plt.plot(y_values_analytical/L_y, u_ex, label="Analytical soln.")
            plt.ylabel(r"$u_x/u_{{max}}$", fontsize=20)
            plt.xlabel(r"$y/L_y$", fontsize=20)
            plt.legend()
            plt.tick_params(direction="in")
    
    
            print("Saving figure to:", os.path.abspath(output), flush=True)
            plt.savefig(output, dpi=400, format='png', bbox_inches='tight')
            
            print("\n\n\n")
    



