import matplotlib as mpl
import pyvista 
import ufl 
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import (
    assemble_vector,
    assemble_matrix,
    create_vector,
    apply_lifting,
    set_bc,
) 
import dolfinx_mpc
import time
import os 
import sys



start_time = time.time() 

# Where to save the plots  
        
    


T = 20
R0 = 2
initDropDiam = 2*R0
L_x = 8*R0
L_y = 2*R0
nx = 80
ny = 30
h = min(L_x/nx, L_y/ny)

beta_mass_diff = 0.000001



Pe = 0.1275 
We = 2
Cn_param=  0.05
theta_deg = 150
dt = (1/10)*Cn_param*Pe*h**2
num_steps = int(np.ceil(T/dt))

Cn = initDropDiam * Cn_param

# Lattice speed of sound
c_s = np.sqrt(1/3)
c_s2 = 1/3



theta = theta_deg * np.pi / 180

WORKDIR = os.getcwd()
outDirName = os.path.join(WORKDIR, "test1") #f"figures_CA{theta_deg}")
os.makedirs(outDirName, exist_ok=True)


tau = 1
xc, yc = L_x/2, R0 - 0.6*R0


# Set up domain. For simplicity, do unit square mesh.


domain = mesh.create_rectangle(MPI.COMM_WORLD,
                               [np.array([0, 0]), np.array([L_x, L_y])],
                               [nx, ny],
                               mesh.CellType.triangle)


# Set periodic boundary conditions at left and right endpoints

Q = 9
# D2Q9 lattice velocities
xi = [
    fem.Constant(domain, (0.0,  0.0)),
    fem.Constant(domain, (1.0,  0.0)),
    fem.Constant(domain, (0.0,  1.0)),
    fem.Constant(domain, (-1.0,  0.0)),
    fem.Constant(domain, (0.0, -1.0)),
    fem.Constant(domain, (1.0,  1.0)),
    fem.Constant(domain, (-1.0,  1.0)),
    fem.Constant(domain, (-1.0, -1.0)),
    fem.Constant(domain, (1.0, -1.0)),
]

# Corresponding weights
w = np.array([
    4/9,
    1/9, 1/9, 1/9, 1/9,
    1/36, 1/36, 1/36, 1/36
])

def left_boundary(x):
    return np.isclose(x[0], 0.0)

def right_boundary(x):
    return np.isclose(x[0], L_x)

def periodic_relation(x):
    out = np.copy(x)
    out[0] = x[0] - L_x
    return out


V = fem.functionspace(domain, ("Lagrange", 1))

pbc = dolfinx_mpc.MultiPointConstraint(V)

# pbc.create_periodic_constraint_geometrical(
#     V,
#     right_boundary,
#     periodic_relation,
#     [],
# )

# pbc.finalize()


# Define trial and test functions, as well as
# finite element functions at previous timesteps

f_trial = ufl.TrialFunction(V)
phi_trial = ufl.TrialFunction(V)
mu_trial = ufl.TrialFunction(V)
f_n = []
for idx in range(Q):
    f_n.append(fem.Function(V))
phi_n = fem.Function(V)
V_vec = fem.functionspace(domain, ("DG", 0, (2,)))
vel_n = fem.Function(V_vec)
mu_n = fem.Function(V)
rho_fn = fem.Function(V)

v = ufl.TestFunction(V)


# Define FE functions to hold post-streaming solution at nP1 timesteps
f_nP1 = []
for idx in range(Q):
    f_nP1.append(fem.Function(V))
phi_nP1 = fem.Function(V)
mu_nP1 = fem.Function(V)

# Define FE functions to hold post-collision distributions
f_star = []
for idx in range(Q):
    f_star.append(fem.Function(V))

force_fn = fem.Function(V_vec)





# Define dynamic pressure
def getDens(f_list):
    return f_list[0] + f_list[1] + f_list[2] + f_list[3] + f_list[4]\
        + f_list[5] + f_list[6] + f_list[7] + f_list[8]


# Define velocity

def getVel(f_list, force_density):
    distr_fn_sum = f_list[0]*xi[0] + f_list[1]*xi[1] + f_list[2]*xi[2]\
        + f_list[3]*xi[3] + f_list[4]*xi[4] + f_list[5]*xi[5]\
        + f_list[6]*xi[6] + f_list[7]*xi[7] + f_list[8]*xi[8]

    density = getDens(f_list)

    vel_term1 = distr_fn_sum/density

    vel_term2 = force_density * dt / (2 * density)

    return vel_term1 + vel_term2


# Define initial equilibrium distributions
def f_equil_init(vel_idx, force_density):
    rho_init = fem.Constant(domain, 1.0)
    rho_expr = fem.Constant(domain, 1.0)

    vel_0 = - (dt/2)*force_density/rho_init

    ci = xi[vel_idx]
    ci_dot_u = ufl.dot(ci, vel_0)
    return w[vel_idx] * rho_expr * (
        1
        + ci_dot_u / c_s**2
        + ci_dot_u**2 / (2*c_s**4)
        - ufl.dot(vel_0, vel_0) / (2*c_s**2)
    )


def f_equil(f_list, vel_idx, force_density):

    rho = getDens(f_list)
    u   = getVel(f_list, force_density)    
    ci       = xi[vel_idx]
    cu = ufl.dot(ci, u)
    u2 = ufl.dot(u, u)

    feq = w[vel_idx] * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)

    

    return feq

    

def body_Force(vel, vel_idx, force_density):
    prefactor = w[vel_idx]
    inverse_cs2 = 1 / c_s**2
    inverse_cs4 = 1 / c_s**4

    xi_dot_prod_F = ufl.dot( xi[vel_idx], force_density)

    u_dot_prod_F = ufl.dot(vel, force_density)

    xi_dot_u = ufl.dot(xi[vel_idx], vel)

    Force = prefactor*(inverse_cs2*(xi_dot_prod_F - u_dot_prod_F)
                       + inverse_cs4*xi_dot_u*xi_dot_prod_F)

    return Force


force_density = -(1/We)*phi_n * ufl.grad(mu_n)


# # Initialize distribution functions. We will use
# where \bar{u}_0 = u_0 - F\Delta t/( 2 \rho_0 ).
# Here we will take u_0 = 0.

for idx in range(Q):
    a = f_trial * v * ufl.dx 
    L = f_equil_init(idx, force_density)*v*ufl.dx
    linProb = dolfinx_mpc.LinearProblem(a, L, mpc=pbc)
    f_n[idx] = linProb.solve() 
    

R0= initDropDiam/2 
eps = Cn
phi_n.interpolate(lambda x: -np.tanh( (np.sqrt(pow(x[0]-xc,2) + pow(x[1]-yc,2)) - R0) / (np.sqrt(2)*eps) ))
mass_diff = fem.Constant(domain, 0.0)

force_density = -(1/We)*phi_n * ufl.grad(mu_n)



# Define boundary conditions.

boundaries = [
    (1, lambda x: np.isclose(x[1], 0)),
    (2, lambda x: np.isclose(x[1], L_y))
    ]


dofs_lower = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[1], 0))

f5_lower = f_n[7]  # rho_expr
f2_lower = f_n[4]  # rho_expr
f6_lower = f_n[8]  # rho_expr

f5_lower_func = fem.Function(V)
f2_lower_func = fem.Function(V)
f6_lower_func = fem.Function(V)

lhs_mat = f_trial*v*ufl.dx
rhs_vec_f5 = f5_lower * v * ufl.dx
linProb_f5 = dolfinx_mpc.LinearProblem(lhs_mat, rhs_vec_f5, mpc=pbc)
f5_lower_func = linProb_f5.solve()

lhs_mat = f_trial*v*ufl.dx
rhs_vec_f2 = f2_lower * v * ufl.dx
linProb_f2 = dolfinx_mpc.LinearProblem(lhs_mat, rhs_vec_f2, mpc=pbc)
f2_lower_func = linProb_f2.solve()

lhs_mat = f_trial*v*ufl.dx
rhs_vec_f6 = f6_lower * v * ufl.dx
linProb_f6 = dolfinx_mpc.LinearProblem(lhs_mat, rhs_vec_f6, mpc=pbc)
f5_lower_func = linProb_f6.solve()

bcs = []

bc_f5 = fem.dirichletbc(f5_lower_func, dofs_lower)
bc_f2 = fem.dirichletbc(f2_lower_func, dofs_lower)
bc_f6 = fem.dirichletbc(f6_lower_func, dofs_lower)

bcs = [bc_f5, bc_f2, bc_f6]

# Similarly, we will define boundary conditions for f_7, f_4, and f_8
# at the upper wall. Once again, boundary conditions simply reduce
# to \rho * w_i


tol = 1e-8

dofs_upper = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[1], L_y))

rho_expr = sum(fk for fk in f_n)

f7_upper = f_n[5]  # rho_expr
f4_upper = f_n[2]  # rho_expr
f8_upper = f_n[6]  # rho_expr

f7_upper_func = fem.Function(V)
f4_upper_func = fem.Function(V)
f8_upper_func = fem.Function(V)

lhs_mat = f_trial*v*ufl.dx

rhs_vec_f7 = f7_upper * v * ufl.dx
linProb_f7 = dolfinx_mpc.LinearProblem(lhs_mat, rhs_vec_f7, mpc=pbc)
f7_upper_func = linProb_f7.solve()

lhs_mat = f_trial*v*ufl.dx
rhs_vec_f4 = f4_upper * v * ufl.dx
linProb_f4 = dolfinx_mpc.LinearProblem(lhs_mat, rhs_vec_f4, mpc=pbc)
f4_upper_func = linProb_f4.solve()

lhs_mat = f_trial*v*ufl.dx
rhs_vec_f8 = f8_upper * v * ufl.dx
linProb_f8 = dolfinx_mpc.LinearProblem(lhs_mat, rhs_vec_f8, mpc=pbc)
f8_upper_func = linProb_f8.solve()

bc_f7 = fem.dirichletbc(f7_upper_func, dofs_upper)
bcs.append(bc_f7)
bc_f4 = fem.dirichletbc(f4_upper_func, dofs_upper)
bcs.append(bc_f4)
bc_f8 = fem.dirichletbc(f8_upper_func, dofs_upper)
bcs.append(bc_f8)


facet_indices, facet_markers = [], []
fdim = domain.topology.dim - 1
for marker, locator in boundaries:
    facets = mesh.locate_entities(domain, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = mesh.meshtags(
    domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets]
)

ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)


# Define variational problems

bilinear_forms_stream = []
linear_forms_stream = []

bilinear_forms_collision = []
linear_forms_collision = []

#n = fem.FacetNormal(domain)
opp_idx = {0: 0, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}


bilin_form_AC = f_trial * v * ufl.dx
bilin_form_mu = f_trial * v * ufl.dx

lin_form_AC = phi_n * v * ufl.dx - dt*v*ufl.dot(getVel(f_n, force_density), ufl.grad(phi_n))*ufl.dx\
    - dt*(1/Pe)*v*mu_n*ufl.dx - (beta_mass_diff/dt)*mass_diff*ufl.sqrt( ufl.dot(ufl.grad(phi_n), ufl.grad(phi_n)) )*v*ufl.dx\
        - 0.5*dt**2 * ufl.dot(getVel(f_n, force_density), ufl.grad(v)) * ufl.dot(getVel(f_n, force_density), ufl.grad(phi_n)) *ufl.dx

lin_form_AC = fem.form(lin_form_AC)

lin_form_mu =  (1/Cn)*( phi_n*(phi_n**2 - 1)*v*ufl.dx\
    + Cn**2*ufl.dot(ufl.grad(phi_n),ufl.grad(v))*ufl.dx\
       - (Cn/(np.sqrt(2)) )*np.cos(theta)*(1 - phi_n**2)*v*ds(1)  )

lin_form_mu = fem.form(lin_form_mu)
    
for idx in range(Q):

    bilinear_forms_collision.append(fem.form(f_trial * v * ufl.dx))
    bilinear_forms_stream.append(fem.form(f_trial * v * ufl.dx))

    double_dot_product_term = -0.5*dt**2 * ufl.dot(xi[idx], ufl.grad(f_star[idx]))\
        * ufl.dot(xi[idx], ufl.grad(v)) * ufl.dx

    dot_product_force_term = 0.5*dt**2 * ufl.dot(xi[idx], ufl.grad(v))\
        * body_Force(getVel(f_n, force_density), idx, force_density) * ufl.dx


    lin_form_idx = f_star[idx]*v*ufl.dx\
        - dt*v*ufl.dot(xi[idx], ufl.grad(f_star[idx]))*ufl.dx\
        + dt*v*body_Force(getVel(f_star, force_density), idx, force_density)*ufl.dx\
        + double_dot_product_term\
        + dot_product_force_term 
        
    lin_form_coll = (f_n[idx] - dt/tau * (f_n[idx] - f_equil(f_n, idx, force_density)) )*v*ufl.dx

    linear_forms_stream.append(fem.form(lin_form_idx))
    linear_forms_collision.append(fem.form(lin_form_coll))

# Assemble matrices for first step

rhs_vec_streaming = [0]*Q
rhs_vec_collision = [0]*Q

sys_mat = []
sys_mat2 = []
for idx in range(Q):
    sys_mat_entry = fem.assemble_matrix(bilinear_forms_stream[idx])
    sys_mat_entry.assemble()
    sys_mat.append(sys_mat_entry)
    
    sys_mat2_entry = fem.assemble_matrix(bilinear_forms_collision[idx])
    sys_mat2_entry.assemble()
    sys_mat2.append(sys_mat2_entry)
    
solver_list = []
solver_list2 = []
for idx in range(Q):
    A = sys_mat[idx]
    A2 = sys_mat2[idx]

    # Create CG solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC.setType(PETSc.PC.Type.LU)
    
    solver2 = PETSc.KSP().create(domain.comm)
    solver2.setOperators(A2)
    solver2.setType(PETSc.KSP.Type.PREONLY)
    solver2.getPC.setType(PETSc.PC.Type.LU)

phi_mat = fem.assemble_matrix(bilin_form_AC)
mu_mat = fem.assemble_matrix(bilin_form_mu)

phi_solver = PETSc.KSP().create(domain.comm)
phi_solver.setOperators(phi_mat)

mu_solver = PETSc.KSP().create(domain.comm)
mu_solver.setOperators(mu_mat)

if 0 == 0:
    log_file = open(outDirName + "/simulation_log.txt", "w")
    log_file.write(f"{'% mass change':>15}"
                   f"{'max ||u||':>15}"
                   f"{'theta':>15}"
                   f"{'smallest f':>15}"
                   f"{'smallest f x':>15}"
                   f"{'smallest f y':>15}"
                   f"{'LB mass':>15}\n")
    log_file.flush()


phi_file = io.XDMFFile(domain.comm, f"{outDirName}/phi.xdmf")
phi_file.parameters["flush_output"] = True
phi_file.parameters["functions_share_mesh"] = True
phi_file.parameters["rewrite_function_mesh"] = False

# mu_file = fe.XDMFFile(comm, f"{outDirName}/mu.xdmf")
# mu_file.parameters["flush_output"] = True
# mu_file.parameters["functions_share_mesh"] = True
# mu_file.parameters["rewrite_function_mesh"] = False

vel_file = io.XDMFFile(domain.comm, f"{outDirName}/vel.xdmf")
vel_file.parameters["flush_output"] = True
vel_file.parameters["functions_share_mesh"] = True
vel_file.parameters["rewrite_function_mesh"] = False

# Timestepping
t = 0.0
mass_init = fem.assemble(phi_n*ufl.dx)
for n in range(num_steps):
    t += dt
    
    #print("n = ", n)
    
    rhs_AC = fem.assemble(lin_form_AC)
    rhs_mu = fem.assemble(lin_form_mu)

    
    # Perform collision, get post-collision distributions f_i^*
    
    for idx in range(Q):
        rhs_vec_collision[idx] = fem.assemble(linear_forms_collision[idx])
        
    for idx in range(Q):
        solver_list2[idx].solve(f_star[idx].vector(), rhs_vec_collision[idx])

    

    # Assemble RHS vectors
    for idx in range(Q):
        rhs_vec_streaming[idx] = (fem.assemble(linear_forms_stream[idx]))

    f5_lower_func.vector()[:] = f_star[7].vector()[:]
    f2_lower_func.vector()[:] = f_star[4].vector()[:]
    f6_lower_func.vector()[:] = f_star[8].vector()[:]
    f7_upper_func.vector()[:] = f_star[5].vector()[:]
    f4_upper_func.vector()[:] = f_star[2].vector()[:]
    f8_upper_func.vector()[:] = f_star[6].vector()[:]

    # # Apply BCs for distribution functions 5, 2, and 6
    bc_f5.apply(sys_mat[5], rhs_vec_streaming[5])
    bc_f2.apply(sys_mat[2], rhs_vec_streaming[2])
    bc_f6.apply(sys_mat[6], rhs_vec_streaming[6])

    # # Apply BCs for distribution functions 7, 4, 8
    bc_f7.apply(sys_mat[7], rhs_vec_streaming[7])
    bc_f4.apply(sys_mat[4], rhs_vec_streaming[4])
    bc_f8.apply(sys_mat[8], rhs_vec_streaming[8])

    # # Solve linear system in each timestep, get f^{n+1}
    for idx in range(Q):
        solver_list[idx].solve(f_nP1[idx].vector(), rhs_vec_streaming[idx])
        
    phi_solver.solve(phi_nP1.vector(), rhs_AC)
    mu_solver.solve(mu_nP1.vector(), rhs_mu)
    


    # Update previous solutions

    for idx in range(Q):
        f_n[idx].assign(f_nP1[idx])
    
    phi_n.assign(phi_nP1)
    mu_n.assign(mu_nP1)
    mass_n = fem.assemble(phi_n*ufl.dx)
    mass_diff.assign( (mass_n - mass_init) )

    distr_dict = {}
    #if fe.MPI.rank(comm) == 0 and os.environ.get("SLURM_PROCID") == "0":
    #if rank == 0:
    if 1 == 1:
        if n % 10== 0:  # plot every 10 steps
        
            #print("n = ", n)

            # rho_expr = sum(fk for fk in f_n)
            # fe.project(rho_expr, V, function=rho_fn)

            # LB_mass = fe.assemble(rho_fn*ufl.dx)
            
            # print("max |drho|   =", np.max(np.abs(rho_diff)))
            # print("max |d_momentum_x|=", np.max(np.abs(momx_diff)))
            # print("max |d_momentum_y|=", np.max(np.abs(momy_diff)))

            # total_mass = fe.assemble(phi_n*ufl.dx)
            # #print("total mass = ", total_mass, flush=True)

            # #print("percent change in mass is ", 100*float(mass_diff)/mass_init, flush=True)

            # percent_mass_change = 100*float(mass_diff)/mass_init

            # vel_expr = getVel(f_n, force_density)
            # fe.project(vel_expr, Vvec, function=vel_n)

            # vel_vec = vel_n.vector().get_local()

            # fe.project(force_density, V_vec, function=force_fn)

            # #force_file.write(force_fn, t)

            # # Determine spatial dimension
            # dim = vel_n.geometric_dimension()

            # # Reshape to (num_nodes, dim)
            # vel_vec = vel_vec.reshape((-1, dim))

            # # Compute nodal norms
            # vel_norm = np.linalg.norm(vel_vec, axis=1)

            # # Maximum nodal value
            # max_vel = vel_norm.max()

            # #print("Max||u||:", max_vel, flush=True)
            
            # for idx in range(Q):
            #     f_vec = f_n[idx].vector().get_local()
            #     min_index = np.argmin(f_vec)
            #     min_value = f_vec[min_index]
                
            #     dof_coords = V.tabulate_dof_coordinates().reshape((-1, V.mesh().geometry().dim()))
            #     min_coord = tuple(dof_coords[min_index])
                
            #     distr_dict[min_coord] = min_value
                
            # min_coord = min(distr_dict, key=distr_dict.get)
            # min_distr = distr_dict[min_coord]

            # theta_avg = 1#computeContactAngle(phi_n, h, Cn, mesh)
                
            # print("theta = ", theta_avg, "\n\n", flush=True)

            # log_file.write(f"{percent_mass_change:15.3f}"
            #                f"{max_vel:15.6e}"
            #                f"{theta_avg:15.2f}"
            #                f"{min_distr:15.3f}"
            #                f"{min_coord[0]:15.2f}"
            #                f"{min_coord[1]:15.2f}"
            #                f"{LB_mass:15.3f} \n")
            # log_file.flush()

            # coords = mesh.coordinates()
            # x = coords[:, 0]   # x-coordinates
            # y  = coords[:, 1]   # y-coordinates
            # phi_vals = phi_n.compute_vertex_values(mesh)
            # triangles = mesh.cells()  # get mesh connectivity
            # triang = tri.Triangulation(coords[:, 0], coords[:, 1], triangles)
        
            # plt.figure(figsize=(6,5))

            # # --- Plot phase field ---
            # plt.tricontourf(triang, phi_vals, levels=50, cmap="RdBu_r")
            # plt.colorbar(label=r"$\phi$")

            # --- Get velocity at vertices ---
            # vel_vertex = vel_n.compute_vertex_values(mesh)
            # dim = mesh.geometry().dim()
            # vel_vertex = vel_vertex.reshape((dim, -1))

            # u_vals = vel_vertex[0, :]
            # v_vals = vel_vertex[1, :]

            # speed = np.sqrt(u_vals**2 + v_vals**2)
            # print("Max velocity =", np.max(speed))
            # print("Min velocity =", np.min(speed))

            # u_vals = u_vals/np.max(speed)
            # v_vals = u_vals/np.max(speed)

            # --- Downsample for clearer quiver plot ---
            # skip = 1   # increase if too many arrows
            # plt.quiver(coords[::skip, 0],
            #             coords[::skip, 1],
            #             u_vals[::skip],
            #             v_vals[::skip],
            #             angles='xy',
            #             width=0.003,          # thicker shafts
            #             headwidth=6,
            #             headlength=7,
            #             color='k')

            # plt.title(f"phi at t = {t:.2f}")
            # plt.xlabel("x")
            # plt.ylabel("y")
            # plt.gca().set_aspect('equal', adjustable='box')
            # plt.tight_layout()

            # out_file = os.path.join(outDirName, f"phi_vel_t{n:05d}.png")
            # plt.savefig(out_file, dpi=200)
            # plt.close()


                        
            # mu_vals = mu_n.compute_vertex_values(mesh)
            # triangles = mesh.cells()  # get mesh connectivity
            # triang = tri.Triangulation(coords[:, 0], coords[:, 1], triangles)
        
            # plt.figure(figsize=(6,5))
            # plt.tricontourf(triang, mu_vals, levels=50, cmap="RdBu_r")
            # plt.colorbar(label=r"$\mu$")
            # plt.title(f"mu at t = {t:.2f}")
            # plt.xlabel("x")
            # plt.ylabel("y")
            # plt.gca().set_aspect('equal', adjustable='box')
            # plt.tight_layout()
            
            # # Save the figure to your output folder
            # out_file = os.path.join(outDirName, f"mu_t{n:05d}.png")
            # plt.savefig(out_file, dpi=200)
            # #plt.show()
            # plt.close()
            
            phi_file.write(phi_n, t)
            vel_file.write(vel_n, t)
            # mu_file.write(mu_n, t)

                