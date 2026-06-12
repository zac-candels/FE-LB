L_phys = 1e-3
nu_phys = 1e-6
l_c = 1e-5
c_s2 = 1/3
tau_kruger = 0.55
tau_duester = 0.1
L_bar = L_phys/l_c
rho_phys = 1000
rho_c = 1000
rho_bar = rho_phys/rho_c
Re = 1250
g = 10

#%% Kruger Method

t_c_kruger = c_s2 * (tau_kruger -1/2)* l_c**2/nu_phys

print("characteristic velocity kruger = ", l_c/t_c_kruger)

a_c_kruger = l_c/t_c_kruger**2 



g_bar = g/a_c_kruger 

F = g_bar

print("force density kruger = ", g_bar)

nu_bar_kruger = (tau_kruger-1/2)*c_s2 

u_max_kruger = Re * nu_bar_kruger / L_bar


print("u_max_kruger = ", u_max_kruger, "\n\n")



#%% Duester Method

print("tau_duester = ", tau_duester)

t_c_duester = c_s2 * (tau_duester )* l_c**2/nu_phys

print("characteristic velocity kruger = ", l_c/t_c_duester)

a_c_duester = l_c/t_c_duester**2 



g_bar = g/a_c_duester

F = g_bar

print("force density = ", g_bar)

nu_bar_duester = tau_duester*c_s2

u_max_duester = F * L_bar**2 / (8*rho_bar * nu_bar_duester)

print("u_max_duester = ", u_max_duester)