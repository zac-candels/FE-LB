import fenics as fe
import os
import numpy as np
#import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from mpi4py import MPI
comm = fe.MPI.comm_world
rank = fe.MPI.rank(comm)

plt.close('all')

# Where to save the plots
WORKDIR = os.getcwd()

def plot_solution_1d(phi, mesh, t, step, outdir):
    """
    Plot and save 1D phase-field solution
    """
    # Get mesh coordinates and solution values
    x = mesh.coordinates().flatten()
    phi_vals = phi.vector().get_local()

    # Sort (important for plotting)
    idx = np.argsort(x)
    x = x[idx]
    phi_vals = phi_vals[idx]

    plt.figure(figsize=(6, 4))
    plt.plot(x, phi_vals, lw=2)
    plt.xlabel("x")
    plt.ylabel(r"$\phi$")
    plt.title(f"t = {t:.4f}")
    plt.ylim(-0.5, 1.5)
    plt.grid(True)

    filename = os.path.join(outdir, f"phi_{step:05d}.png")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

T = 100
L_x, L_y = 1, 1
nx, ny = 128, 128
h = L_x/nx
dt = 0.1*h
num_steps = int(np.ceil(T/dt))

M_tilde = 0.005

outDirName = f"1D"

if rank == 0:
    os.makedirs(outDirName, exist_ok=True)
comm.Barrier()


eps = 0.7*h

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

# Define trial and test functions, as well as
# finite element functions at previous timesteps

f_trial = fe.TrialFunction(V)

phi_n = fe.Function(V)
mu_nP1 = fe.Function(V)

v = fe.TestFunction(V)

phi_nP1 = fe.Function(V)


# Define Allen-Cahn mobility
def mobility(phi_n):
    grad_phi_n = fe.grad(phi_n)

    abs_grad_phi_n = fe.sqrt(fe.dot(grad_phi_n, grad_phi_n) + 1e-6)
    inv_abs_grad_phi_n = 1.0 / abs_grad_phi_n

    mob = M_tilde*( 1 - 4*phi_n*(1 - phi_n)/eps * inv_abs_grad_phi_n )
    return mob


# Initialize \phi
phi_init_expr = fe.Expression(
    "0.5 * (1.0 + tanh((x[0] - x0)/(sqrt(2)*eps)))",
    degree=4,
    eps=eps,
    x0=0.0
)


phi_n = fe.project(phi_init_expr, V)

bilin_form_AC = f_trial * v * fe.dx

lin_form_AC = phi_n * v * fe.dx - dt*fe.dot(fe.grad(v), mobility(phi_n)*fe.grad(phi_n))*fe.dx


phi_mat = fe.assemble(bilin_form_AC)

phi_solver = fe.KrylovSolver("cg", "hypre_amg")
phi_solver.set_operator(phi_mat)
prm = phi_solver.parameters
prm["absolute_tolerance"] = 1e-14
prm["relative_tolerance"] = 1e-8
prm["maximum_iterations"] = 1000
prm["nonzero_initial_guess"] = False

outfile = fe.XDMFFile(comm, f"{outDirName}/solution.xdmf")
outfile.parameters["flush_output"] = True
outfile.parameters["functions_share_mesh"] = True
outfile.parameters["rewrite_function_mesh"] = False

# Timestepping
t = 0.0
for n in range(num_steps):
    t += dt
    
    if n % 10 == 0:  # plot every 10 steps
    
        #c_n.rename("phi", "phase field")  # Optional: label for visualization
        #xdmf_file.write(c_n, t)
        
        outfile.write(phi_n, t)

        if rank == 0:
            plot_solution_1d(phi_n, mesh, t, n, outDirName)
            
    
    rhs_AC = fe.assemble(lin_form_AC)
    
    phi_solver.solve(phi_nP1.vector(), rhs_AC)


    # Update previous solutions

    phi_n.assign(phi_nP1)
            
    
    a = 1    


    
    









    
    






