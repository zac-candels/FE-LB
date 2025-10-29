import numpy as np

T = 1500
CFL = 0.2
d = 1
L_x, L_y = 9*d, 9*d
nx = ny = 100
h = L_x/nx
dt = h*CFL
num_steps = int(np.ceil(T/dt))


g = 9.81

# Density of heavier phase
rho_h = 0.001
# Lattice speed of sound
c_s2 = 1/3

# Bond number
Bo = 100

# Morton number
Mo = 1000

# Cahn number
Cn = 0.05

eps = Cn * d

sigma = g*rho_h*d**2/Bo

eta_h = (Mo * sigma**3 * rho_h)**(1/4)
eta_l = eta_h/100

beta = 12*sigma/eps

kappa = 3*sigma*eps/2 

# Relaxation times for heavier and lighter phases
tau_h = eta_h / (c_s2 * rho_h * dt )
tau_l =  eta_l / (c_s2 * rho_h * dt )

