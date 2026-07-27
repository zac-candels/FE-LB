import sys
sys.path.insert(0, "/home/zcandels/refactor/src")
import lattice
import meshAndFnSpaces
from postProcessing import writeData
import finiteElementFunctions
import moments
import streamingModule
import collision
import testMod
import fenics as fe
import os
import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from petsc4py import PETSc
#import numba as nb
import shutil
import json


def main():

    start_time = time.time()
    comm = fe.MPI.comm_world
    rank = fe.MPI.rank(comm)
    plt.close('all')
    
    with open("params.json") as file:
        params = json.load(file)

    dim = params["dim"]
    T = params["Tfinal"]
    L_x = params["L_x"]
    L_y = params["L_y"]
    nx = params["nx"]
    ny = params["ny"]
    forceDensityTuple = params["Force_density"]
    Force_density = fe.Constant(forceDensityTuple)
    # Lattice speed of sound
    c_s = params["c_s"]
    # Number of discrete velocities
    Q = params["Q"]
    # Density on wall
    rho_wall = params["rho_wall"]
    # Initial density
    rho_init = params["rho_init"]
    u_wall = params["u_wall"]
    u_max = params["u_max"]
    tau = params["tau"]
    lumping = params["lumping"]
    forceInCollisionStreaming = params["forceInCollisionStreaming"]
    # D2Q9 lattice velocities
    
    latticeClass = lattice.D2Q9()
    xi = latticeClass.xi
    w = latticeClass.weights
    xi_arr = latticeClass.xi_arr
    
    mesh, V, Vvec = meshAndFnSpaces.create_mesh(dim, L_x, L_y, nx, ny)


    h = mesh.hmin()
    dt = 0.005*h/np.sqrt(2)
    num_steps = int(np.ceil(T/dt))

    outDirName = writeData.create_output_directory(dt, h, name="refactor")
    
 
    simState = finiteElementFunctions.SimulationState(V, Vvec, Q)
    
    forceDensity_x = fe.Function(V)
    forceDensity_y = fe.Function(V)
    
    # Define velocity
    

    
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
    
    
    
    # # Initialize distribution functions. We will use
    # f_i^{0} \gets f_i^{0, eq}( \rho_0, \bar{u}_0 ),
    # where \bar{u}_0 = u_0 - F\Delta t/( 2 \rho_0 ).
    # Here we will take u_0 = 0.
    for idx in range(Q):
        simState.f_n[idx] = (fe.project(f_equil_init(idx, Force_density), V))
    
    
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
        


    # # Similarly, we will define boundary conditions for f_7, f_4, and f_8
    # # at the upper wall. Here, the conjugate distributions are 
    # # 5, 2, and 6. 
    tol = 1e-8
    def Bdy_Upper(x, on_boundary):
        if on_boundary:
            if fe.near(x[1], L_y, tol):
                return True
            else:
                return False
        else:
            return False

    
    streamer = streamingModule.StreamingOperator(V,
                                                 simState,
                                                 latticeClass,
                                                 dt,
                                                 lumping,
                                                 forceInCollisionStreaming)
    
    lower_pairs = [(5,7), (2,4), (6,8)]
    upper_pairs = [(7,5), (4,2), (8,6)]
    upper_bcs = testMod.BounceBackBoundary(V, streamer, simState.f_n,
                                              Bdy_Lower, upper_pairs)
    lower_bcs = testMod.BounceBackBoundary(V, streamer, simState.f_n,
                                              Bdy_Upper, lower_pairs)
    
    vel_file = fe.XDMFFile(comm, f"{outDirName}/vel.xdmf")
    vel_file.parameters["flush_output"] = True
    vel_file.parameters["functions_share_mesh"] = True
    vel_file.parameters["rewrite_function_mesh"] = False
    
    
    
    forceVec_x = simState.f_star[0].vector().copy()
    forceVec_y = simState.f_star[0].vector().copy()
        
    

        
    # Timestepping
    t = 0.0
    forceVals_x = []
    forceVals_y = []
    for n in range(num_steps):
        t += dt
        
        pre_coll_time = time.time()
        # We will try to do collision locally, since it is a pure
        # time-dependnet ODE
        
        fe.assemble(Force_density.values()[0]*simState.v*fe.dx, tensor=forceVec_x )
        fe.assemble(simState.v*fe.dx, tensor=forceVec_y)
        forceVec_y.vec().scale(0)
        
        fe.solve(streamer.massMat, forceDensity_x.vector(), forceVec_x)
        # petscForce_x = fe.as_backend_type(forceVec_x)
        # forceDensity_x.vector().vec().pointwiseDivide(petscForce_x.vec(), M_petsc)
        fe.solve(streamer.massMat, forceDensity_y.vector(), forceVec_y)
        # petscForce_y = fe.as_backend_type(forceVec_y)
        # forceDensity_y.vector().vec().pointwiseDivide(petscForce_y.vec(), M_petsc)
        projectForceTimeEnd = time.time()
        #print("project force time = ", projectForceTimeEnd - projectForceTimeStart)
        
        pre_coll_time_lb = time.time()
        # We will try to do collision locally, since it is a pure
        # time-dependnet ODE
        
        forceVals_x = forceDensity_x.vector().get_local()
        #forceVals_x = forceVals_x.reshape((-1, mesh.geometry().dim()))
        
        forceVals_y = forceDensity_y.vector().get_local()
        #forceVals_y = forceVals_y.reshape((-1, mesh.geometry().dim()))
    
        simState.f_star = collision.collideLocal(simState.f_n, 
                                       simState.f_star, 
                                       latticeClass,
                                       (forceVals_x, forceVals_y),
                                       tau,
                                       dt)
    
        #print("collision_time =", post_coll_time - pre_coll_time)
        
    
        pre_stream_time = time.time()
        # Assemble RHS vectors for streaming step
        
        streamer.assembleRhsLumping(simState.f_star, dt)
        #print("stream assemble =", post_assemble_stream_time - pre_stream_time)
        
        
    
    
        lower_bcs.update(simState.f_star)
        upper_bcs.update(simState.f_star)
    
        lower_bcs.applyRhsVec(streamer.rhsVecStreaming)
        upper_bcs.applyRhsVec(streamer.rhsVecStreaming)
    
        pre_stream_time = time.time()
        # Solve linear system for streaming step
        
        simState.f_nP1 = streamer.solveSysLumping(simState.f_nP1)
       
        lower_bcs.applyF_nP1(simState.f_nP1)
        upper_bcs.applyF_nP1(simState.f_nP1)
        
        #print("time to solve stream sys ", post_stream_time - pre_stream_time, "\n\n\n\n")
    
    
        # Update previous solutions
    
        for idx in range(Q):
            simState.f_n[idx].assign(simState.f_nP1[idx])
            

    
        if n % 10000 == 0:
            print("n = ", n)
            vel_expr = moments.getVel(simState.f_n, xi_arr, forceDensityTuple, dt)
            fe.project(vel_expr, Vvec, function=simState.vel_n)
            vel_file.write(simState.vel_n, t)
            u_new, v_new = 0, 0
            
            for i in range(Q):
                xi_new = xi[i].values()
                u_new += simState.f_n[i].vector().get_local()*xi_new[0]
                v_new += simState.f_n[i].vector().get_local()*xi_new[1]
    
            u_e = fe.Expression('u_max*( 1 - pow( (2*x[1]/L_y -1), 2 ) )',
                                degree=2, u_max=u_max, L_y=L_y)
            u_e = fe.interpolate(u_e, V)
            error = np.linalg.norm(u_e.vector().get_local() - u_new)
            time_elapsed = time.time() - start_time
            print('t = %.4f: error = %.3g' % (t, error), flush=True)
            print('max u:', u_new.max(), flush=True)
            print("Time elapsed = ", time_elapsed, "\n\n", flush=True)
    
            num_points_analytical = 200
            num_points_numerical = 10
            y_values_analytical = np.linspace(0, L_y, num_points_analytical)
            y_values_numerical = np.linspace(0, L_y, num_points_numerical)
            x_fixed = L_x/2
            points = [(x_fixed, y) for y in y_values_numerical]
            u_x_values = []
            u_ex = np.linspace(0, L_y, num_points_analytical)
            nu = tau/3
            u_max = Force_density.values()[0]*L_y**2/(8*rho_init*nu)
            for i in range(num_points_analytical):
                u_ex[i] = (1 - (2*y_values_analytical[i]/L_y - 1)**2)
    
            for point in points:
                u_at_point = simState.vel_n(point)
                u_x_values.append(u_at_point[0] / u_max)
    
    
    
            fig_name = "felb_dt" + str(dt) + "_simTime" + str(n) + ".png"
            output = os.path.join(outDirName, fig_name)
    
            plt.figure()
            plt.plot(y_values_numerical/L_y, u_x_values, 'o', label="FE soln.")
            plt.plot(y_values_analytical/L_y, u_ex, label="Analytical soln.")
            plt.ylabel(r"$u_x/u_{{max}}$", fontsize=20)
            plt.xlabel(r"$y/L_y$", fontsize=20)
            plt.legend()
            plt.tick_params(direction="in")
    
    
            print("Saving figure to:", os.path.abspath(output))
            plt.savefig(output, dpi=400, format='png', bbox_inches='tight')
            

main()
