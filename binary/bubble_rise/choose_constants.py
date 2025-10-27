import numpy as np

CFL = 0.2
d = 1
L_x, L_y = 9*d, 9*d
nx = ny = 100
h = L_x/nx
dt = h*CFL


g = 9.81
rho_h = 0.001
c_s2 = 1/3

Bo = 100

Mo = 1000

Cn = 0.05

xi = Cn * d

sigma = g*rho_h*d**2/Bo

eta_h = (Mo * sigma**3 * rho_h)**(1/3)
eta_l = eta_h/100

beta = 12*sigma/xi

kappa = 3*sigma*xi/2 

tau_h = eta_h / (c_s2 * rho_h * dt )
tau_l =  eta_l / (c_s2 * rho_h * dt )