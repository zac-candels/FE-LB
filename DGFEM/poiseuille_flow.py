import fenics as fe
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

T = 3000
dt = 1
num_steps = int(np.ceil(T/dt))


Re = 0.96
nx = ny = 20
L_x = 10
L_y = 20
h = L_x/nx

error_vec = []

# Lattice speed of sound
c_s = np.sqrt(1/3) # np.sqrt( 1./3. * h**2/dt**2 )

#nu = 1.0/6.0
#tau = nu/c_s**2 + dt/2 
tau = 1

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


nu = tau/3
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

# Set up domain. For simplicity, do unit square mesh.

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


V = fe.FunctionSpace(mesh, "DG", 1, constrained_domain=pbc)




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


advection_forms = []
advection_mats = []
face_forms = []
mass_forms = []
mass_mats = []
n = fe.FacetNormal(mesh)

for i in range(Q):
    mass_forms.append( f_trial[i] * v * fe.dx )
    mass_mats.append( fe.assemble(mass_forms[i]))
    
    xi_i = xi[i]
    xi_vec = fe.as_vector( (xi_i[0], xi_i[1]) )

    # Element interior term: - ∫ f (c·∇phi) dx
    interior_contribution = -fe.dot(xi_vec, fe.grad(v) )*f_trial[i]*fe.dx

    # Upwind flux on interior facets:
    # Upwind trace: when dot(c,n('+')) >= 0 -> use u('+') else u('-')
    # The weak DG flux contribution for test function phi uses jump operator.
    # We use average normal since facet measure is symmetric. Implementation with conditional:
    # For interior facets (dS):
    #    flux = dot(ci,n('+')) * (conditional(dot(ci,n('+'))>=0, f_trial('+'), f_trial('-')))*phi('+') ...
    # Simpler and stable implementation: use upwind term in symmetric form
    #    ∫_{facets} |c·n| * (u_up)*jump(phi) ds
    # We'll construct a practical upwind form:
    lam = fe.dot(xi_vec, n)
    # upwind trace: use conditional on lam
    f_up = fe.conditional(lam('+') >= 0, f_trial('+'), f_trial('-'))
    # The flux contribution to the weak form:
    facet_contribution = (lam('+')*f_up*v('+'))*fe.dS + (lam('-')*f_trial*v)*fe.ds  # ds adds boundary flux using exterior f_trial

    a_form = interior_contribution + facet_contribution
    advection_forms.append(a_form)
    A = fe.assemble(fe.lhs(fe.assemble(a_form + 0*v*fe.dx)))  # trick to get matrix? Instead, assemble bilinear form properly
    # Proper assembly of operator matrix: bilinear form b(u,v) = a_form with TrialFunction as u and TestFunction as v
    b = fe.replace(a_form, {f_trial: fe.TrialFunction(V), v: fe.TestFunction(V)})  # ensure fresh
    advection_mat = fe.assemble(b)
    advection_mats.append(advection_mat)
    
    
for timesteps:
    
    # For each direction assemble RHS = -A f + M * s(f)
    for i in range(Q):
        # advection term
        Af = A_mats[i]*f[i].vector()
        # collision/source: s_i = -(1/tau)*(f_i - feq_i)
        feq_i_proj = project(feq_list[i], V)  # project eq to DG space
        s_vec = feq_i_proj.vector()
        s_vec *= (1.0 / tau)
        s_vec *= -1.0
        s_vec.axpy(1.0 / tau, f[i].vector())  # -(1/tau)*(f - feq) = -(1/tau)f + (1/tau)feq -> combine properly
        # But above ended with wrong sign; compute properly:
        # compute rhs_vec = -A f + M * (-(1/tau)*(f - feq))
        rhs = Af.copy()
        rhs *= -1.0
        # M * s where s = -(1/tau)*(f - feq) -> compute s_vec = -(1/tau)*(f - feq)
        s_local = f[i].vector().copy()
        s_local.axpy(-1.0, feq_i_proj.vector())  # s_local = f - feq
        s_local *= -(1.0 / tau)                   # s_local = -(1/tau)*(f - feq)
        # Add M*s_local to rhs
        Ms = M * s_local
        rhs.axpy(1.0, Ms)

        # Multiply by dt and apply M^{-1} to get update
        rhs *= dt
        # Solve M * delta = rhs  -> delta = M^{-1} rhs
        delta = rhs.copy()
        M_solver.solve(delta, rhs)  # delta = M^{-1} rhs
        # Update f_new = f + delta
        f_new[i].vector()[:] = f[i].vector() + delta

    # Swap f and f_new for next step
    for i in range(Q):
        f[i].vector()[:] = f_new[i].vector()
    
    
    
    














