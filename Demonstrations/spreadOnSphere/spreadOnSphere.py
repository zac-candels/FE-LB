import sys
sys.path.insert(0, "/home/zcandels/FE-LB")
import fenics as fe
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import time 
import mshr
import shutil
from scipy      import optimize
import src.postProcessing.computeContactAngle as cca 
 
comm = fe.MPI.comm_world
rank = fe.MPI.rank(comm)

start_time = time.time() 

plt.close('all')

# Where to save the plots
fe.set_log_active(False)
fe.parameters["form_compiler"]["optimize"] = True
fe.parameters["form_compiler"]["cpp_optimize"] = True
      

T = 300
R0 = 1
initDropDiam = 2*R0
L_x = 4*R0
L_y = 6*R0
nx = 50
ny = 70
h = min(L_x/nx, L_y/ ny)

A = 0.5
kappa = 0.02
interfaceThickness = np.sqrt(kappa/A)
tau = 0.1
M_tilde = 10
theta_deg = 30


# Lattice speed of sound
c_s = np.sqrt(1/3)
c_s2 = 1/3



theta = theta_deg * np.pi / 180

WORKDIR = os.getcwd()
outDirName = os.path.join(WORKDIR, f"spreadOnSphere")
if os.path.exists(outDirName):
    shutil.rmtree(outDirName)
os.makedirs(outDirName, exist_ok=True)


xc, yc = L_x/2, L_y/2 + L_y/4

Q = 9
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

rectangle = mshr.Rectangle(fe.Point(0.0, 0.0), fe.Point(L_x, L_y))

# Circular hole in the center
radius = R0
center = fe.Point(L_x/2, L_y/2)
circle = mshr.Circle(fe.Point(L_x/2, L_y/2), radius)

# Subtract circle from rectangle
domain = rectangle - circle

# Generate mesh
mesh = mshr.generate_mesh(domain, 70)

boundary_markers = fe.MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)

tol = mesh.hmin()/10
ctr_circle_facet = 0
ctr_bdy_marker = 0 
for facet in fe.facets(mesh):
    
    x = facet.midpoint()
    
    r = np.sqrt((x.x() - center.x())**2 + (x.y() - center.y())**2)

    if abs(r - radius) < tol:
        print("circle boundary facet")
        ctr_circle_facet +=1
        n = facet.normal()
        
        # Bottom boundary → normal has negative y component
        if n.y() < 0 and n.x() < 0 and abs(n.x()) < abs(n.y()):
            print("boundary marker 1 upperHalf_downSlope_lt45")
            boundary_markers[facet] = 1
            ctr_bdy_marker+=1
            
        if n.y() < 0 and n.x() > 0 and abs(n.x()) > abs(n.y()):
            print("boundary marker 2 upperHalf_downSlope_gt45")
            boundary_markers[facet] = 2
            ctr_bdy_marker+=1
            
        if n.y() > 0 and n.x() < 0 and abs(n.x()) > abs(n.y()):
            print("boundary marker 3 lowerHalf_upSlope_gt45")
            boundary_markers[facet] = 3
            ctr_bdy_marker+=1
            
        if n.y() > 0 and n.x() < 0 and abs(n.x()) < abs(n.y()):
            print("boundary marker 4 lowerHalf_upSlope_lt45")
            boundary_markers[facet] = 4
            ctr_bdy_marker+=1
            
        if n.y() > 0 and n.x() > 0 and abs(n.x()) < abs(n.y()):
            print("boundary marker 5 lowerHalf_downSlope_lt45")
            boundary_markers[facet] = 5
            ctr_bdy_marker+=1
            
        if n.y() > 0 and n.x() > 0 and abs(n.x()) > abs(n.y()):
            print("boundary marker 6 lowerHalf_downSlope_gt45")
            boundary_markers[facet] = 6
            ctr_bdy_marker+=1
            
        if n.y() < 0 and n.x() > 0 and abs(n.x()) > abs(n.y()):
            print("boundary marker 7 upperHalf_upSlope_gt45")
            boundary_markers[facet] = 7
            ctr_bdy_marker+=1
            
        if n.y() < 0 and n.x() > 0 and abs(n.x()) < abs(n.y()):
            print("boundary marker 8 upperHalf_upSlope_lt45")
            boundary_markers[facet] = 8
            ctr_bdy_marker+=1

print("total number of circle facets is", ctr_circle_facet )
print("total number of boundary markers is ", ctr_bdy_marker)
h = mesh.hmin()
dt = 0.005*h**2
#dt = 0.0001
beta_mass_diff =  0.1*dt
num_steps = int(np.ceil(T/dt))
# Set periodic boundary conditions at left and right endpoints


class PeriodicBoundary(fe.SubDomain):
    def inside(self, x, on_boundary):
        return bool(x[0] < fe.DOLFIN_EPS and x[0] > -fe.DOLFIN_EPS and on_boundary)
    def map(self, x, y):
        y[0] = x[0] - L_x
        y[1] = x[1]


pbc = PeriodicBoundary()


V = fe.FunctionSpace(mesh, "Lagrange", 1, constrained_domain=pbc)
dofCoords = V.tabulate_dof_coordinates()
dofCoords = dofCoords.reshape((-1, mesh.geometry().dim()))


# Define trial and test functions, as well as
# finite element functions at previous timesteps

f_trial = fe.TrialFunction(V)
phi_trial = fe.TrialFunction(V)
mu_trial = fe.TrialFunction(V)
f_n = []
for idx in range(Q):
    f_n.append(fe.Function(V))
phi_n = fe.Function(V)
V_cont = fe.VectorFunctionSpace(mesh, "P", 1, constrained_domain=pbc)
V_dis = fe.VectorFunctionSpace(mesh, "DG", 0, constrained_domain=pbc)

vel_star = fe.Function(V_cont)
vel_cont = fe.Function(V_cont)
vel_dis = fe.Function(V_dis)
mu_n = fe.Function(V)
rho_n = fe.Function(V)
forceDensity_x = fe.Function(V)
forceDensity_y = fe.Function(V)

v = fe.TestFunction(V)
v_vec = fe.TestFunction(V_cont)
trial_vec = fe.TrialFunction(V_cont)

# Define FE functions to hold post-streaming solution at nP1 timesteps
f_nP1 = []
for idx in range(Q):
    f_nP1.append(fe.Function(V))
phi_nP1 = fe.Function(V)
mu_nP1 = fe.Function(V)

# Define FE functions to hold post-collision distributions
f_star = []
for idx in range(Q):
    f_star.append(fe.Function(V))

force_fn = fe.Function(V_dis)




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
    rho_init = fe.Constant(1.0)
    rho_expr = fe.Constant(1.0)

    vel_0 = - (dt/2)*force_density/rho_init
    
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
    
    F_u_outer1 = fe.outer( force_density, vel_0 )
    u_F_outer2 = fe.outer(vel_0, force_density) 
    force_vel_outer = F_u_outer1 + u_F_outer2
    c_dot_F = fe.inner( ci, force_density)
    
    f_neq = - w[vel_idx]*tau/c_s2 * rho_expr * fe.inner(Q, vel_grad)\
        - w[vel_idx]*dt/(2*c_s2) * ( c_dot_F\
                                 + 1/(2*c_s2) * fe.inner(Q, force_vel_outer) )
    
    
    return f_eq + f_neq


xi_array = np.array([[float(c.values()[0]), float(c.values()[1])] for c in xi])


force_density = -phi_n * fe.grad(mu_n)


# # Initialize distribution functions. We will use
# where \bar{u}_0 = u_0 - F\Delta t/( 2 \rho_0 ).
# Here we will take u_0 = 0.

for idx in range(Q):
    f_n[idx] = (fe.project(f_equil_init(idx, force_density), V))
    
# Initialize \phi
c_init_expr = fe.Expression(
    "-tanh( (sqrt(pow(x[0]-xc,2) + pow(x[1]-yc,2)) - R) / (sqrt(2)*eps) )",
    degree=2,  # polynomial degree used for interpolation
    xc=xc,
    yc=yc,
    R=initDropDiam/2,
    eps=interfaceThickness
)

phi_n = fe.interpolate(c_init_expr, V)
mass_diff = fe.Constant(0.0)

force_density = -phi_n * fe.grad(mu_n)

forceDensity_n = fe.project(force_density, V_dis)

# Define boundary conditions.

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

# Boundary conditions for lowerHalf_upSlope_lt45 boundary marker 4
f7_lowerHalf_upSlope_lt45 = f_n[5] 
f4_lowerHalf_upSlope_lt45 = f_n[2] 
f8_lowerHalf_upSlope_lt45 = f_n[6] 
f1_lowerHalf_upSlope_lt45 = f_n[3]

f7_lowerHalf_upSlope_lt45_func = fe.Function(V)
f4_lowerHalf_upSlope_lt45_func = fe.Function(V)
f8_lowerHalf_upSlope_lt45_func = fe.Function(V)
f1_lowerHalf_upSlope_lt45_func = fe.Function(V)

fe.project(f7_lowerHalf_upSlope_lt45, V, function=f7_lowerHalf_upSlope_lt45_func)
fe.project(f4_lowerHalf_upSlope_lt45, V, function=f4_lowerHalf_upSlope_lt45_func)
fe.project(f8_lowerHalf_upSlope_lt45, V, function=f8_lowerHalf_upSlope_lt45_func)
fe.project(f1_lowerHalf_upSlope_lt45, V, function=f1_lowerHalf_upSlope_lt45_func)

bc_f7_lowerHalf_upSlope_lt45 = fe.DirichletBC(V, f7_lowerHalf_upSlope_lt45_func, boundary_markers, 4)
bc_f4_lowerHalf_upSlope_lt45 = fe.DirichletBC(V, f4_lowerHalf_upSlope_lt45_func, boundary_markers, 4)
bc_f8_lowerHalf_upSlope_lt45 = fe.DirichletBC(V, f8_lowerHalf_upSlope_lt45_func, boundary_markers, 4)
bc_f1_lowerHalf_upSlope_lt45 = fe.DirichletBC(V, f1_lowerHalf_upSlope_lt45_func, boundary_markers, 4)


# Boundary conditions for lowerHalf_upSlope_gt45 boundary marker 3
f4_lowerHalf_upSlope_gt45 = f_n[2] 
f8_lowerHalf_upSlope_gt45 = f_n[6] 
f1_lowerHalf_upSlope_gt45 = f_n[3] 
f5_lowerHalf_upSlope_gt45 = f_n[7]

f4_lowerHalf_upSlope_gt45_func = fe.Function(V)
f8_lowerHalf_upSlope_gt45_func = fe.Function(V)
f1_lowerHalf_upSlope_gt45_func = fe.Function(V)
f5_lowerHalf_upSlope_gt45_func = fe.Function(V)

fe.project(f4_lowerHalf_upSlope_gt45, V, function=f4_lowerHalf_upSlope_gt45_func)
fe.project(f8_lowerHalf_upSlope_gt45, V, function=f8_lowerHalf_upSlope_gt45_func)
fe.project(f1_lowerHalf_upSlope_gt45, V, function=f1_lowerHalf_upSlope_gt45_func)
fe.project(f5_lowerHalf_upSlope_gt45, V, function=f5_lowerHalf_upSlope_gt45_func)

bc_f4_lowerHalf_upSlope_gt45 = fe.DirichletBC(V, f4_lowerHalf_upSlope_gt45_func, boundary_markers, 3)
bc_f8_lowerHalf_upSlope_gt45 = fe.DirichletBC(V, f8_lowerHalf_upSlope_gt45_func, boundary_markers, 3)
bc_f1_lowerHalf_upSlope_gt45 = fe.DirichletBC(V, f1_lowerHalf_upSlope_gt45_func, boundary_markers, 3)
bc_f5_lowerHalf_upSlope_gt45 = fe.DirichletBC(V, f5_lowerHalf_upSlope_gt45_func, boundary_markers, 3)


# Boundary conditions for lowerHalf_downSlope_gt45 boundary marker 6
f6_lowerHalf_downSlope_gt45 = f_n[8] 
f3_lowerHalf_downSlope_gt45 = f_n[1] 
f7_lowerHalf_downSlope_gt45 = f_n[5] 
f4_lowerHalf_downSlope_gt45 = f_n[2]

f6_lowerHalf_downSlope_gt45_func = fe.Function(V)
f3_lowerHalf_downSlope_gt45_func = fe.Function(V)
f7_lowerHalf_downSlope_gt45_func = fe.Function(V)
f4_lowerHalf_downSlope_gt45_func = fe.Function(V)

fe.project(f6_lowerHalf_downSlope_gt45, V, function=f6_lowerHalf_downSlope_gt45_func)
fe.project(f3_lowerHalf_downSlope_gt45, V, function=f3_lowerHalf_downSlope_gt45_func)
fe.project(f7_lowerHalf_downSlope_gt45, V, function=f7_lowerHalf_downSlope_gt45_func)
fe.project(f4_lowerHalf_downSlope_gt45, V, function=f4_lowerHalf_downSlope_gt45_func)

bc_f6_lowerHalf_downSlope_gt45 = fe.DirichletBC(V, f6_lowerHalf_downSlope_gt45_func, boundary_markers, 6)
bc_f3_lowerHalf_downSlope_gt45 = fe.DirichletBC(V, f3_lowerHalf_downSlope_gt45_func, boundary_markers, 6)
bc_f7_lowerHalf_downSlope_gt45 = fe.DirichletBC(V, f7_lowerHalf_downSlope_gt45_func, boundary_markers, 6)
bc_f4_lowerHalf_downSlope_gt45 = fe.DirichletBC(V, f4_lowerHalf_downSlope_gt45_func, boundary_markers, 6)


# Boundary conditions for lowerHalf_downSlope_lt45 boundary marker 5
f3_lowerHalf_downSlope_lt45 = f_n[1]
f7_lowerHalf_downSlope_lt45 = f_n[5] 
f4_lowerHalf_downSlope_lt45 = f_n[2] 
f8_lowerHalf_downSlope_lt45 = f_n[6] 

f3_lowerHalf_downSlope_lt45_func = fe.Function(V)
f7_lowerHalf_downSlope_lt45_func = fe.Function(V)
f4_lowerHalf_downSlope_lt45_func = fe.Function(V)
f8_lowerHalf_downSlope_lt45_func = fe.Function(V)

fe.project(f3_lowerHalf_downSlope_lt45, V, function=f3_lowerHalf_downSlope_lt45_func)
fe.project(f7_lowerHalf_downSlope_lt45, V, function=f7_lowerHalf_downSlope_lt45_func)
fe.project(f4_lowerHalf_downSlope_lt45, V, function=f4_lowerHalf_downSlope_lt45_func)
fe.project(f8_lowerHalf_downSlope_lt45, V, function=f8_lowerHalf_downSlope_lt45_func)

bc_f3_lowerHalf_downSlope_lt45 = fe.DirichletBC(V, f3_lowerHalf_downSlope_lt45_func, boundary_markers, 5)
bc_f7_lowerHalf_downSlope_lt45 = fe.DirichletBC(V, f7_lowerHalf_downSlope_lt45_func, boundary_markers, 5)
bc_f4_lowerHalf_downSlope_lt45 = fe.DirichletBC(V, f4_lowerHalf_downSlope_lt45_func, boundary_markers, 5)
bc_f8_lowerHalf_downSlope_lt45 = fe.DirichletBC(V, f8_lowerHalf_downSlope_lt45_func, boundary_markers, 5)


# Boundary conditions for upperHalf_upSlope_lt45 boundary marker 8
f5_upperHalf_upSlope_lt45 = f_n[7] 
f2_upperHalf_upSlope_lt45 = f_n[4] 
f6_upperHalf_upSlope_lt45 = f_n[8] 
f3_upperHalf_upSlope_lt45 = f_n[1]

f5_upperHalf_upSlope_lt45_func = fe.Function(V)
f2_upperHalf_upSlope_lt45_func = fe.Function(V)
f6_upperHalf_upSlope_lt45_func = fe.Function(V)
f3_upperHalf_upSlope_lt45_func = fe.Function(V)

fe.project(f5_upperHalf_upSlope_lt45, V, function=f5_upperHalf_upSlope_lt45_func)
fe.project(f2_upperHalf_upSlope_lt45, V, function=f2_upperHalf_upSlope_lt45_func)
fe.project(f6_upperHalf_upSlope_lt45, V, function=f6_upperHalf_upSlope_lt45_func)
fe.project(f3_upperHalf_upSlope_lt45, V, function=f3_upperHalf_upSlope_lt45_func)

bc_f5_upperHalf_upSlope_lt45 = fe.DirichletBC(V, f5_upperHalf_upSlope_lt45_func, boundary_markers, 8)
bc_f2_upperHalf_upSlope_lt45 = fe.DirichletBC(V, f2_upperHalf_upSlope_lt45_func, boundary_markers, 8)
bc_f6_upperHalf_upSlope_lt45 = fe.DirichletBC(V, f6_upperHalf_upSlope_lt45_func, boundary_markers, 8)
bc_f3_upperHalf_upSlope_lt45 = fe.DirichletBC(V, f3_upperHalf_upSlope_lt45_func, boundary_markers, 8)

# Boundary conditions for upperHalf_upSlope_gt45 boundary marker 7
f2_upperHalf_upSlope_gt45 = f_n[4] 
f6_upperHalf_upSlope_gt45 = f_n[8] 
f3_upperHalf_upSlope_gt45 = f_n[1] 
f7_upperHalf_upSlope_gt45 = f_n[5]

f2_upperHalf_upSlope_gt45_func = fe.Function(V)
f6_upperHalf_upSlope_gt45_func = fe.Function(V)
f3_upperHalf_upSlope_gt45_func = fe.Function(V)
f7_upperHalf_upSlope_gt45_func = fe.Function(V)

fe.project(f2_upperHalf_upSlope_gt45, V, function=f2_upperHalf_upSlope_gt45_func)
fe.project(f6_upperHalf_upSlope_gt45, V, function=f6_upperHalf_upSlope_gt45_func)
fe.project(f3_upperHalf_upSlope_gt45, V, function=f3_upperHalf_upSlope_gt45_func)
fe.project(f7_upperHalf_upSlope_gt45, V, function=f7_upperHalf_upSlope_gt45_func)

bc_f2_upperHalf_upSlope_gt45 = fe.DirichletBC(V, f2_upperHalf_upSlope_gt45_func, boundary_markers, 7)
bc_f6_upperHalf_upSlope_gt45 = fe.DirichletBC(V, f6_upperHalf_upSlope_gt45_func, boundary_markers, 7)
bc_f3_upperHalf_upSlope_gt45 = fe.DirichletBC(V, f3_upperHalf_upSlope_gt45_func, boundary_markers, 7)
bc_f7_upperHalf_upSlope_gt45 = fe.DirichletBC(V, f7_upperHalf_upSlope_gt45_func, boundary_markers, 7)


# Boundary conditions for upperHalf_downSlope_gt45 boundary marker 2
f8_upperHalf_downSlope_gt45 = f_n[6] 
f1_upperHalf_downSlope_gt45 = f_n[3] 
f5_upperHalf_downSlope_gt45 = f_n[7] 
f2_upperHalf_downSlope_gt45 = f_n[4]

f8_upperHalf_downSlope_gt45_func = fe.Function(V)
f1_upperHalf_downSlope_gt45_func = fe.Function(V)
f5_upperHalf_downSlope_gt45_func = fe.Function(V)
f2_upperHalf_downSlope_gt45_func = fe.Function(V)

fe.project(f8_upperHalf_downSlope_gt45, V, function=f8_upperHalf_downSlope_gt45_func)
fe.project(f1_upperHalf_downSlope_gt45, V, function=f1_upperHalf_downSlope_gt45_func)
fe.project(f5_upperHalf_downSlope_gt45, V, function=f5_upperHalf_downSlope_gt45_func)
fe.project(f2_upperHalf_downSlope_gt45, V, function=f2_upperHalf_downSlope_gt45_func)

bc_f8_upperHalf_downSlope_gt45 = fe.DirichletBC(V, f8_upperHalf_downSlope_gt45_func, boundary_markers, 2)
bc_f1_upperHalf_downSlope_gt45 = fe.DirichletBC(V, f1_upperHalf_downSlope_gt45_func, boundary_markers, 2)
bc_f5_upperHalf_downSlope_gt45 = fe.DirichletBC(V, f5_upperHalf_downSlope_gt45_func, boundary_markers, 2)
bc_f2_upperHalf_downSlope_gt45 = fe.DirichletBC(V, f2_upperHalf_downSlope_gt45_func, boundary_markers, 2)


# Boundary conditions for upperHalf_downSlope_lt45 boundary marker 1
f1_upperHalf_downSlope_lt45 = f_n[3] 
f5_upperHalf_downSlope_lt45 = f_n[7] 
f2_upperHalf_downSlope_lt45 = f_n[4] 
f6_upperHalf_downSlope_lt45 = f_n[8]

f1_upperHalf_downSlope_lt45_func = fe.Function(V)
f5_upperHalf_downSlope_lt45_func = fe.Function(V)
f2_upperHalf_downSlope_lt45_func = fe.Function(V)
f6_upperHalf_downSlope_lt45_func = fe.Function(V)

fe.project(f1_upperHalf_downSlope_lt45, V, function=f1_upperHalf_downSlope_lt45_func)
fe.project(f5_upperHalf_downSlope_lt45, V, function=f5_upperHalf_downSlope_lt45_func)
fe.project(f2_upperHalf_downSlope_lt45, V, function=f2_upperHalf_downSlope_lt45_func)
fe.project(f6_upperHalf_downSlope_lt45, V, function=f6_upperHalf_downSlope_lt45_func)

bc_f1_upperHalf_downSlope_lt45 = fe.DirichletBC(V, f1_upperHalf_downSlope_lt45_func, boundary_markers, 1)
bc_f5_upperHalf_downSlope_lt45 = fe.DirichletBC(V, f5_upperHalf_downSlope_lt45_func, boundary_markers, 1)
bc_f2_upperHalf_downSlope_lt45 = fe.DirichletBC(V, f2_upperHalf_downSlope_lt45_func, boundary_markers, 1)
bc_f6_upperHalf_downSlope_lt45 = fe.DirichletBC(V, f6_upperHalf_downSlope_lt45_func, boundary_markers, 1)


cells_with_marker1 = []

for cell in fe.cells(mesh):

    for facet in fe.facets(cell):

        if boundary_markers[facet] == 1:

            mp = cell.midpoint()

            cells_with_marker1.append((mp.x(), mp.y()))

            break   # avoid printing the same cell multiple times


print("Number of cells:", len(cells_with_marker1))

for x, y in cells_with_marker1:
    print(f"({x:.6f}, {y:.6f})")

# Define variational problems

bilinear_forms_stream = []
linear_forms_stream = []

bilinear_forms_collision = []
linear_forms_collision = []

# Create MeshFunction for boundary markers
boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)

# Subdomain for bottom wall
class Bottom(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and fe.near(x[1], 0.0)


ds = fe.Measure("ds", domain=mesh, subdomain_data=boundary_markers)
ds_bottom = ds(1) + ds(2) + ds(3) + ds(4) + ds(5) + ds(6) + ds(7) + ds(8)

bilin_form_AC = f_trial * v * fe.dx
bilin_form_mu = f_trial * v * fe.dx


lin_form_AC = - dt*v*fe.dot(getVel(f_n, force_density), fe.grad(phi_n))*fe.dx\
    - dt*M_tilde*v*mu_n*fe.dx - (beta_mass_diff/dt)*mass_diff*fe.sqrt( fe.dot(fe.grad(phi_n), fe.grad(phi_n)) )*v*fe.dx\
        - 0.5*dt**2 * fe.dot(getVel(f_n, force_density), fe.grad(v)) * fe.dot(getVel(f_n, force_density), fe.grad(phi_n)) *fe.dx

lin_form_mu =  A*phi_n**3*v*fe.dx\
    + kappa*fe.dot(fe.grad(phi_n),fe.grad(v))*fe.dx\
       + kappa/(np.sqrt(2)*interfaceThickness)*np.cos(theta)*(phi_n**2-1)*v*ds_bottom
       
advection_forms = []
double_advection_forms = []

# Define linear and bilinear forms for the collision and streaming steps
for idx in range(Q):

    bilinear_forms_stream.append(f_trial * v * fe.dx)
    
    advection_forms.append( v*fe.dot(xi[idx], fe.grad(f_trial))*fe.dx )
    double_advection_forms.append( fe.dot(xi[idx],fe.grad(v))\
                                  *fe.dot(xi[idx],fe.grad(f_trial))*fe.dx )


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
prevTimeMuVec = f_star[0].vector().copy()
rhsVecACTemp = f_star[0].vector().copy()
rhsVecMuTemp = f_star[0].vector().copy()

xi_arr = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],
                   [1,1],[-1,1],[-1,-1],[1,-1]], dtype=float)

phi_mat = fe.assemble(bilin_form_AC)
mu_mat = fe.assemble(bilin_form_mu)
phi_solver = fe.KrylovSolver("cg", "hypre_amg")
phi_solver.set_operator(phi_mat)

mu_solver = fe.KrylovSolver("cg", "hypre_amg")
mu_solver.set_operator(mu_mat)

rhs_AC = fe.assemble(lin_form_AC)
rhs_mu = fe.assemble(lin_form_mu)

forceVec_x = rhs_mu.copy()
forceVec_y = rhs_mu.copy()

if 1==1:
    log_file = open(outDirName + "/simulation_log.txt", "w")
    log_file.write(f"{'n' :>15}"
                   f"{'% mass change':>15}"
                   f"{'max ||u||':>15}"
                   f"{'theta':>15}"
                   f"{'smallest f':>15}"
                   f"{'smallest f x':>15}"
                   f"{'smallest f y':>15}"
                   f"{'LB mass':>15}\n")
    log_file.flush()


phi_file = fe.XDMFFile(comm, f"{outDirName}/phi.xdmf")
phi_file.parameters["flush_output"] = True
phi_file.parameters["functions_share_mesh"] = True
phi_file.parameters["rewrite_function_mesh"] = False

mu_file = fe.XDMFFile(comm, f"{outDirName}/mu.xdmf")
mu_file.parameters["flush_output"] = True
mu_file.parameters["functions_share_mesh"] = True
mu_file.parameters["rewrite_function_mesh"] = False

vel_file = fe.XDMFFile(comm, f"{outDirName}/vel.xdmf")
vel_file.parameters["flush_output"] = True
vel_file.parameters["functions_share_mesh"] = True
vel_file.parameters["rewrite_function_mesh"] = False

div_file = fe.XDMFFile(comm, f"{outDirName}/div.xdmf")
div_file.parameters["flush_output"] = True
div_file.parameters["functions_share_mesh"] = True
div_file.parameters["rewrite_function_mesh"] = False

pres_file = fe.XDMFFile(comm, f"{outDirName}/pres.xdmf")
pres_file.parameters["flush_output"] = True 
pres_file.parameters["functions_share_mesh"] = True 
pres_file.parameters["rewrite_function_mesh"] = False

# # Apply BCs for upSlope boundary
# bc_f5_upSlope.apply(sys_mat[5])
# bc_f2_upSlope.apply(sys_mat[2])
# bc_f6_upSlope.apply(sys_mat[6])
# bc_f3_upSlope.apply(sys_mat[3])

# # Apply BCs for downSlope boundary
# bc_f1_downSlope.apply(sys_mat[1])
# bc_f5_downSlope.apply(sys_mat[5])
# bc_f2_downSlope.apply(sys_mat[2])
# bc_f6_downSlope.apply(sys_mat[6])

# # Apply BCs for top boundary
# bc_f7_upper.apply(sys_mat[7])
# bc_f4_upper.apply(sys_mat[4])
# bc_f8_upper.apply(sys_mat[8])

xi_arr = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],
                   [1,1],[-1,1],[-1,-1],[1,-1]], dtype=float)

dof_coords = V.tabulate_dof_coordinates().reshape((-1, 2))
wall_dofs = np.where(
    (np.abs(dof_coords[:, 1]) < 1e-10) |
    (np.abs(dof_coords[:, 1] - L_y) < 1e-10)
)[0]
# Timestepping
t = 0.0
forceVals_x = []
forceVals_y = []
inverse_cs2 = 1 / c_s**2
inverse_cs4 = 1 / c_s**4
mass_init = fe.assemble( (phi_n+1)/2*fe.dx)

facet_vis = fe.MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)

for facet in fe.facets(mesh):
    if facet.exterior() and boundary_markers[facet] in boundary_markers:
        facet_vis[facet] = 1

fe.File("bottom_facets.pvd") << facet_vis

phi_file.write(phi_n, t)
for n in range(num_steps):
    t += dt
    
    #print("n = ", n)
    prevTimeAcVec.zero()
    fe.as_backend_type(prevTimeAcVec).vec().pointwiseMult(phi_n.vector().vec(), sysMatLumped[0])
    fe.assemble(lin_form_AC, tensor=rhsVecACTemp)
    rhs_AC = prevTimeAcVec + rhsVecACTemp
    
    prevTimeMuVec.zero()
    fe.as_backend_type(prevTimeMuVec).vec().pointwiseMult(-A*phi_n.vector().vec(), sysMatLumped[0])
    fe.assemble(lin_form_mu, tensor=rhsVecMuTemp)
    rhs_mu = prevTimeMuVec + rhsVecMuTemp
    
    pre_coll_time_lb = time.time()
    # We will try to do collision locally, since it is a pure
    # time-dependnet ODE
    
    f_vals = np.array([f_n[idx].vector().get_local() for idx in range(Q)])
    
    
    fe.assemble(-phi_n * fe.grad(mu_n)[0]*v*fe.dx, tensor=forceVec_x )
    fe.assemble(-phi_n * fe.grad(mu_n)[1]*v*fe.dx, tensor=forceVec_y)
    
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
    ux[wall_dofs] = 0.0
    uy[wall_dofs] = 0.0
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

    u_dot_prod_F = fe.dot(vel_star, force_density)
    
    stream_FE_start_time = time.time()
    for idx in range(Q):
        
        if idx == 0:
            Force = -w[idx]*(inverse_cs2* u_dot_prod_F)

        else:
    
            xi_dot_prod_F = fe.dot( xi[idx], force_density)
    
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
        rhsVecStreaming[idx].axpy(0.5*dt**2, doubleAdvectionVecs[idx])
        
        rhsVecStreaming[idx].axpy(dt, basicForceTerm)
        rhsVecStreaming[idx].axpy(0.5*dt**2, advectionForceTerm)
    stream_FE_end_time = time.time()
    #print("stream FE time = ", stream_FE_end_time - stream_FE_start_time)
    
    # f5_noSlope_func.vector()[:] = f_star[7].vector()[:]
    # f2_noSlope_func.vector()[:] = f_star[4].vector()[:]
    # f6_noSlope_func.vector()[:] = f_star[8].vector()[:]

    f7_upper_func.assign(f_star[5] )
    f4_upper_func.assign(f_star[2] )
    f8_upper_func.assign(f_star[6] )
    
    f5_lower_func.assign(f_star[7])
    f2_lower_func.assign(f_star[4])
    f6_lower_func.assign(f_star[8])
    
    assignApplyTimeStart = time.time()
    f7_lowerHalf_upSlope_lt45_func.assign(f_star[5])
    f4_lowerHalf_upSlope_lt45_func.assign(f_star[2])
    f8_lowerHalf_upSlope_lt45_func.assign(f_star[6])
    f1_lowerHalf_upSlope_lt45_func.assign(f_star[3])
    
    f4_lowerHalf_upSlope_gt45_func.assign(f_star[2])
    f8_lowerHalf_upSlope_gt45_func.assign(f_star[6])
    f1_lowerHalf_upSlope_gt45_func.assign(f_star[3])
    f5_lowerHalf_upSlope_gt45_func.assign(f_star[7])
    
    f6_lowerHalf_downSlope_gt45_func.assign(f_star[8])
    f3_lowerHalf_downSlope_gt45_func.assign(f_star[1])
    f7_lowerHalf_downSlope_gt45_func.assign(f_star[5])
    f4_lowerHalf_downSlope_gt45_func.assign(f_star[2])
    
    f3_lowerHalf_downSlope_lt45_func.assign(f_star[1])
    f7_lowerHalf_downSlope_lt45_func.assign(f_star[5])
    f4_lowerHalf_downSlope_lt45_func.assign(f_star[2])
    f8_lowerHalf_downSlope_lt45_func.assign(f_star[6])
    
    f5_upperHalf_upSlope_lt45_func.assign(f_star[7])
    f2_upperHalf_upSlope_lt45_func.assign(f_star[4])
    f6_upperHalf_upSlope_lt45_func.assign(f_star[8])
    f3_upperHalf_upSlope_lt45_func.assign(f_star[1])
    
    f2_upperHalf_upSlope_gt45_func.assign(f_star[4])
    f6_upperHalf_upSlope_gt45_func.assign(f_star[8])
    f3_upperHalf_upSlope_gt45_func.assign(f_star[1])
    f7_upperHalf_upSlope_gt45_func.assign(f_star[5])
    
    f8_upperHalf_downSlope_gt45_func.assign(f_star[6])
    f1_upperHalf_downSlope_gt45_func.assign(f_star[3])
    f5_upperHalf_downSlope_gt45_func.assign(f_star[7])
    f2_upperHalf_downSlope_gt45_func.assign(f_star[4])
    
    f1_upperHalf_downSlope_lt45_func.assign(f_star[3])
    f5_upperHalf_downSlope_lt45_func.assign(f_star[7])
    f2_upperHalf_downSlope_lt45_func.assign(f_star[4])
    f6_upperHalf_downSlope_lt45_func.assign(f_star[8])
    
    bc_f7_lowerHalf_upSlope_lt45.apply(rhsVecStreaming[7])
    bc_f4_lowerHalf_upSlope_lt45.apply(rhsVecStreaming[4])
    bc_f8_lowerHalf_upSlope_lt45.apply(rhsVecStreaming[8])
    bc_f1_lowerHalf_upSlope_lt45.apply(rhsVecStreaming[1])
    
    bc_f4_lowerHalf_upSlope_gt45.apply(rhsVecStreaming[4])
    bc_f8_lowerHalf_upSlope_gt45.apply(rhsVecStreaming[8])
    bc_f1_lowerHalf_upSlope_gt45.apply(rhsVecStreaming[1])
    bc_f5_lowerHalf_upSlope_gt45.apply(rhsVecStreaming[5])
    
    bc_f6_lowerHalf_downSlope_gt45.apply(rhsVecStreaming[6])
    bc_f3_lowerHalf_downSlope_gt45.apply(rhsVecStreaming[3])
    bc_f7_lowerHalf_downSlope_gt45.apply(rhsVecStreaming[7])
    bc_f4_lowerHalf_downSlope_gt45.apply(rhsVecStreaming[4])
    
    bc_f3_lowerHalf_downSlope_lt45.apply(rhsVecStreaming[3])
    bc_f7_lowerHalf_downSlope_lt45.apply(rhsVecStreaming[7])
    bc_f4_lowerHalf_downSlope_lt45.apply(rhsVecStreaming[4])
    bc_f8_lowerHalf_downSlope_lt45.apply(rhsVecStreaming[8])
    
    bc_f5_upperHalf_upSlope_lt45.apply(rhsVecStreaming[5])
    bc_f2_upperHalf_upSlope_lt45.apply(rhsVecStreaming[2])
    bc_f6_upperHalf_upSlope_lt45.apply(rhsVecStreaming[6])
    bc_f3_upperHalf_upSlope_lt45.apply(rhsVecStreaming[3])
    
    bc_f2_upperHalf_upSlope_gt45.apply(rhsVecStreaming[2])
    bc_f6_upperHalf_upSlope_gt45.apply(rhsVecStreaming[6])
    bc_f3_upperHalf_upSlope_gt45.apply(rhsVecStreaming[3])
    bc_f7_upperHalf_upSlope_gt45.apply(rhsVecStreaming[7])
    
    bc_f8_upperHalf_downSlope_gt45.apply(rhsVecStreaming[8])
    bc_f1_upperHalf_downSlope_gt45.apply(rhsVecStreaming[1])
    bc_f5_upperHalf_downSlope_gt45.apply(rhsVecStreaming[5])
    bc_f2_upperHalf_downSlope_gt45.apply(rhsVecStreaming[2])
    
    bc_f1_upperHalf_downSlope_lt45.apply(rhsVecStreaming[1])
    bc_f5_upperHalf_downSlope_lt45.apply(rhsVecStreaming[5])
    bc_f2_upperHalf_downSlope_lt45.apply(rhsVecStreaming[2])
    bc_f6_upperHalf_downSlope_lt45.apply(rhsVecStreaming[6])
    # # Apply BCs for distribution functions 5, 2, and 6
    # bc_f5.apply( rhsVecStreaming[5])
    # bc_f2.apply( rhsVecStreaming[2])
    # bc_f6.apply( rhsVecStreaming[6])

    # # # Apply BCs for distribution functions 7, 4, 8
    # bc_f7.apply( rhsVecStreaming[7])
    # bc_f4.apply(rhsVecStreaming[4])
    # bc_f8.apply(rhsVecStreaming[8])

    solveTimeStart = time.time()
    # # Solve linear system in each timestep, get f^{n+1}
    for idx in range(Q):
        #solver_list[idx].solve(f_nP1[idx].vector(), rhsVecStreaming[idx])
        vi = fe.as_backend_type(rhsVecStreaming[idx]).vec()
        f_nP1[idx].vector().vec().pointwiseDivide(vi, sysMatLumped[idx])
        
    bc_f7_lowerHalf_upSlope_lt45.apply(f_nP1[7].vector())
    bc_f4_lowerHalf_upSlope_lt45.apply(f_nP1[4].vector())
    bc_f8_lowerHalf_upSlope_lt45.apply(f_nP1[8].vector())
    bc_f1_lowerHalf_upSlope_lt45.apply(f_nP1[1].vector())
    
    bc_f4_lowerHalf_upSlope_gt45.apply(f_nP1[4].vector())
    bc_f8_lowerHalf_upSlope_gt45.apply(f_nP1[8].vector())
    bc_f1_lowerHalf_upSlope_gt45.apply(f_nP1[1].vector())
    bc_f5_lowerHalf_upSlope_gt45.apply(f_nP1[5].vector())
    
    bc_f6_lowerHalf_downSlope_gt45.apply(f_nP1[6].vector())
    bc_f3_lowerHalf_downSlope_gt45.apply(f_nP1[3].vector())
    bc_f7_lowerHalf_downSlope_gt45.apply(f_nP1[7].vector())
    bc_f4_lowerHalf_downSlope_gt45.apply(f_nP1[4].vector())
    
    bc_f3_lowerHalf_downSlope_lt45.apply(f_nP1[3].vector())
    bc_f7_lowerHalf_downSlope_lt45.apply(f_nP1[7].vector())
    bc_f4_lowerHalf_downSlope_lt45.apply(f_nP1[4].vector())
    bc_f8_lowerHalf_downSlope_lt45.apply(f_nP1[8].vector())
    
    bc_f5_upperHalf_upSlope_lt45.apply(f_nP1[5].vector())
    bc_f2_upperHalf_upSlope_lt45.apply(f_nP1[2].vector())
    bc_f6_upperHalf_upSlope_lt45.apply(f_nP1[6].vector())
    bc_f3_upperHalf_upSlope_lt45.apply(f_nP1[3].vector())
    
    bc_f2_upperHalf_upSlope_gt45.apply(f_nP1[2].vector())
    bc_f6_upperHalf_upSlope_gt45.apply(f_nP1[6].vector())
    bc_f3_upperHalf_upSlope_gt45.apply(f_nP1[3].vector())
    bc_f7_upperHalf_upSlope_gt45.apply(f_nP1[7].vector())
    
    bc_f8_upperHalf_downSlope_gt45.apply(f_nP1[8].vector())
    bc_f1_upperHalf_downSlope_gt45.apply(f_nP1[1].vector())
    bc_f5_upperHalf_downSlope_gt45.apply(f_nP1[5].vector())
    bc_f2_upperHalf_downSlope_gt45.apply(f_nP1[2].vector())
    
    bc_f1_upperHalf_downSlope_lt45.apply(f_nP1[1].vector())
    bc_f5_upperHalf_downSlope_lt45.apply(f_nP1[5].vector())
    bc_f2_upperHalf_downSlope_lt45.apply(f_nP1[2].vector())
    bc_f6_upperHalf_downSlope_lt45.apply(f_nP1[6].vector())
        
    # Apply BCs for lower boundary
    bc_f5.apply( f_nP1[5].vector())
    bc_f2.apply(f_nP1[2].vector())
    bc_f6.apply( f_nP1[6].vector())
    
    # Apply BCs for top boundary
    bc_f7.apply( f_nP1[7].vector())
    bc_f4.apply( f_nP1[4].vector())
    bc_f8.apply( f_nP1[8].vector())
        
    #phi_solver.solve(phi_nP1.vector(), rhs_AC)
    rhsPhiVec = fe.as_backend_type(rhs_AC).vec()
    phi_nP1.vector().vec().pointwiseDivide(rhsPhiVec, sysMatLumped[0])
    rhsMuVec = fe.as_backend_type(rhs_mu).vec()
    #mu_solver.solve(mu_nP1.vector(), rhs_mu)
    mu_nP1.vector().vec().pointwiseDivide(rhsMuVec, sysMatLumped[0])

    solveTimeEnd = time.time()
    #print("solve time = ", solveTimeEnd - solveTimeStart)
    # Update previous solutions

    for idx in range(Q):
        f_n[idx].assign(f_nP1[idx])
    
    # mass_n = fe.assemble(phi_n*fe.dx) 
    # mass_nP1 = fe.assemble(phi_nP1*fe.dx)
    # mass_diff.assign( ( mass_nP1 - mass_n ) )
    
    # print("mass_n = ", mass_n)
    # print("mass_nP1 = ", mass_nP1)
    # print("mass_diff = ", mass_diff.values()[0])
    
    phi_n.assign(phi_nP1)
    mu_n.assign(mu_nP1)
    
    mass_n = fe.assemble( (phi_n+1)/2*fe.dx)
    mass_diff.assign( (mass_n - mass_init) )
    

    

    distr_dict = {}
    #if rank == 0:
    #if fe.MPI.rank(comm) == 0 and os.environ.get("SLURM_PROCID") == "0":
    if n < 40000000:
        if n % 100== 0:  # plot every 10 steps
            print("n = ", n)
            
            
            rho_expr = getDens(f_n)
            fe.project(rho_expr, V, function=rho_n)
            
            vel_expr = getVel(f_n, force_density)
            #fe.project(vel_expr, V_dis, function=vel_dis)
            fe.project(vel_expr, V_cont, function=vel_cont)
            #div_u = fe.project(fe.div(vel_cont), V)
            iteration_time = time.time()
            print("time elapsed ", iteration_time - start_time, "\n")
            phi_file.write(phi_n, t)
            vel_file.write(vel_cont, t)
            pres_file.write(rho_n, t)
            #div_file.write(div_u, t)
            #mu_file.write(mu_n, t)
            

            
            #print("mass_n = ", mass_nP1)
            #massDiffNonLinTerm = fe.assemble(fe.sqrt( fe.dot(fe.grad(phi_n), fe.grad(phi_n)) )*v*fe.dx)
            print("phi max = ", np.max(phi_n.vector().get_local()))
            print("phi min = ", np.min(phi_n.vector().get_local()))
            
            #print("gradPhi norm = ", np.linalg.norm(.vector().get_local()))
            percent_mass_change = 100*float(mass_diff)/mass_init
            print("mass change = ", percent_mass_change, "%")
            
            # Determine spatial dimension
            dim = vel_cont.geometric_dimension()
            vel_vec = vel_cont.vector().get_local()
            # Reshape to (num_nodes, dim)
            vel_vec = vel_vec.reshape((-1, dim))

            # Compute nodal norms
            vel_norm = np.linalg.norm(vel_vec, axis=1)
        

            # Maximum nodal value
            max_vel = vel_norm.max()
            print("umax = ", max_vel)
            for idx in range(Q):
                f_vec = f_n[idx].vector().get_local()
                min_index = np.argmin(f_vec)
                min_value = f_vec[min_index]
                
                dof_coords = V.tabulate_dof_coordinates().reshape((-1, V.mesh().geometry().dim()))
                min_coord = tuple(dof_coords[min_index])
                
                distr_dict[min_coord] = min_value
                
            min_coord = min(distr_dict, key=distr_dict.get)
            min_distr = distr_dict[min_coord]
            
            rho_vals = rho_n.vector().get_local()
            print("max density is", np.max(rho_vals))
            print("min density is", np.min(rho_vals))

            LB_mass = fe.assemble(rho_n*fe.dx)
            
            theta_avg = 1#cca.computeContactAngle_gradPhi(phi_n, h, interfaceThickness, mesh)
            theta_geom = 1#cca.computeContactAngle_heightDiam(phi_n, h, interfaceThickness, mesh)
                
            print("theta avg = ", theta_avg, flush=True)
            print("theta geom = ", theta_geom, "\n\n", flush=True)

            log_file.write(f"{n:15d}"
                           f"{percent_mass_change:15.3f}"
                           f"{max_vel:15.8g}"
                           f"{theta_avg:15.2f}"
                           f"{min_distr:15.3f}"
                           f"{min_coord[0]:15.2f}"
                           f"{min_coord[1]:15.2f}"
                           f"{LB_mass:15.3f} \n")
            log_file.flush()
            

if rank == 0:
    log_file.close()
                