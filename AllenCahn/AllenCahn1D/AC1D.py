import numpy as np
import fenics as fe
import matplotlib.pyplot as plt

dt = 1e-4
T = 1
c1 = np.sqrt(0.0001)
c2 = 5

num_steps = int(T/dt)

class PeriodicBoundary(fe.SubDomain):
    # Left boundary is "target"
    def inside(self, x, on_boundary):
        return bool(on_boundary and fe.near(x[0], -1.0))

    # Map right boundary (x=1) to left (x=-1)
    def map(self, x, y):
        if fe.near(x[0], 1.0):
            y[0] = x[0] - 1.0
        else:
            y[0] = x[0]

pbc = PeriodicBoundary()

mesh = fe.IntervalMesh(100, -1, 1)

V = fe.FunctionSpace(mesh, "CG", 1, constrained_domain=pbc)


phi_nP1 = fe.Function(V)
v = fe.TestFunction(V)

phi_init_expr = fe.Expression("pow(x[0],2)  * cos(pi * x[0])",
                              pi = 3.14159, degree = 2)

phi_n = fe.interpolate(phi_init_expr, V)


Functional = (phi_nP1 - phi_n)*v*fe.dx + dt*c1**2 * fe.dot( fe.grad(phi_nP1), fe.grad(v) )*fe.dx\
    + dt*c2*( phi_nP1**3 - phi_nP1 )*v*fe.dx
    

J = fe.derivative(Functional, phi_nP1, fe.TrialFunction(V))



problem = fe.NonlinearVariationalProblem(Functional, phi_nP1, bcs=None, J=J)
solver  = fe.NonlinearVariationalSolver(problem)
prm = solver.parameters
# optional: configure Newton tolerances
prm['newton_solver']['relative_tolerance'] = 1e-8
prm['newton_solver']['maximum_iterations'] = 25
prm['newton_solver']['linear_solver'] = 'mumps'  # or 'lu' / 'cg' + precond


t = 0
for n in range(num_steps):
    
    t += dt
    
    solver.solve()
    
    phi_n.assign(phi_nP1)
    
    if abs(t - 0.75) < dt:
        plt.figure
        fe.plot(phi_nP1)
        title_str = f"t= {t}"
        plt.title(title_str)
        
        
        a = 1
    
    
    
    





