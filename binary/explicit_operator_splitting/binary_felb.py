import fenics as fe 
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

T = 3000
dt = 1
num_steps = int(np.ceil(T/dt))


Re = 0.96
nx = ny = 5
L_x = 30
L_y = 30
h = L_x/nx

# Lattice speed of sound
c_s = np.sqrt(1/3) # np.sqrt( 1./3. * h**2/dt**2 )

#nu = 1.0/6.0
#tau = nu/c_s**2 + dt/2 
tau_ns = 1
tau_ch = 0.1

# Number of discrete velocities
Q = 9
Force_density = np.array([2.6041666e-5, 0.0])

# Density on wall
rho_wall = 1.0
# Initial density 
rho_init = 1.0
u_wall = (0.0, 0.0)

# Liquid density
rho_l = 1.0
# Air density
rho_g = 0.001

# Mobility
M = 0.2

radius = 10
drop_center_x = L_x / 2
drop_center_y = 2


nu = tau_ch/3
u_max = Force_density[0]*L_y**2/(8*rho_init*nu)


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

# Set up domain mesh.
mesh = fe.RectangleMesh(fe.Point(0,0), fe.Point(L_x, L_y), nx, nx )

# Set periodic boundary conditions at left and right endpoints
class PeriodicBoundaryX(fe.SubDomain):
    def inside(self, x, on_boundary):
        return fe.near(x[0], 0.0) and on_boundary

    def map(self, x, y):
        # Map left boundary to the right
        y[0] = x[0] - L_x
        y[1] = x[1]

pbc = PeriodicBoundaryX()

# Create function spaces
V = fe.FunctionSpace(mesh, "P", 1, constrained_domain=pbc)
V_vec = fe.VectorFunctionSpace(mesh, "P", constrained_domain=pbc)

# Create trial and test functions for weak forms

trial_fn = fe.TrialFunction(V)
v = fe.TestFunction(V)


# Define dynamic pressure
def dynPres_fn(g_list):
    return g_list[0] + g_list[1] + g_list[2] + g_list[3] + g_list[4]\
        + g_list[5] + g_list[6] + g_list[7] + g_list[8]
        
# Define order parameter
def orderParam_fn(h_list):
    return h_list[0] + h_list[1] + h_list[2] + h_list[3] + h_list[4]\
        + h_list[5] + h_list[6] + h_list[7] + h_list[8]
        


        
