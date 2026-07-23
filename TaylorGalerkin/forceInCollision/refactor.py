import sys
sys.path.insert(0, "/home/zcandels/refactor/src")
import lattice
import meshAndFnSpaces
from postProcessing import writeData
import finiteElementFunctions
import moments
import streamingModule
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
    
    f5_lower = simState.f_n[7]  
    f2_lower = simState.f_n[4] 
    f6_lower = simState.f_n[8] 
    
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
    # at the upper wall. Here, the conjugate distributions are 
    # 5, 2, and 6. 
    tol = 1e-8
    def Bdy_Upper(x, on_boundary):
        if on_boundary:
            if fe.near(x[1], L_y, tol):
                return True
            else:
                return False
        else:
            return False
    
    f7_upper = simState.f_n[5]  
    f4_upper = simState.f_n[2]  
    f8_upper = simState.f_n[6]  
    
    f7_upper_func = fe.Function(V)
    f4_upper_func = fe.Function(V)
    f8_upper_func = fe.Function(V)
    
    fe.project(f7_upper, V, function=f7_upper_func)
    fe.project(f4_upper, V, function=f4_upper_func)
    fe.project(f8_upper, V, function=f8_upper_func)
    
    bc_f7 = fe.DirichletBC(V, f7_upper_func, Bdy_Upper)
    bc_f4 = fe.DirichletBC(V, f4_upper_func, Bdy_Upper)
    bc_f8 = fe.DirichletBC(V, f8_upper_func, Bdy_Upper)
    
    streamer = streamingModule.StreamingOperator(V,
                                                 simState,
                                                 latticeClass,
                                                 dt)
    
    vel_file = fe.XDMFFile(comm, f"{outDirName}/vel.xdmf")
    vel_file.parameters["flush_output"] = True
    vel_file.parameters["functions_share_mesh"] = True
    vel_file.parameters["rewrite_function_mesh"] = False
    
    
    # Apply BCs to matrices for distribution functions 5, 2, and 6
    bc_f5.apply(streamer.sysMatStream[5])
    #bc_f5.apply(fe.PETScVector(sysMatLumped[5]))
    bc_f5.apply(streamer.advectionMats[5])
    bc_f5.apply(streamer.doubleAdvectionMats[5])
    
    bc_f2.apply(streamer.sysMatStream[2])
    #bc_f2.apply(fe.PETScVector(sysMatLumped[2]))
    bc_f2.apply(streamer.advectionMats[2])
    bc_f2.apply(streamer.doubleAdvectionMats[2])
    
    bc_f6.apply(streamer.sysMatStream[6])
    #bc_f6.apply(fe.PETScVector(sysMatLumped[6]))
    bc_f6.apply(streamer.advectionMats[6])
    bc_f6.apply(streamer.doubleAdvectionMats[6])
    
    # Apply BCs to matrices for distribution functions 7, 4, 8
    bc_f7.apply(streamer.sysMatStream[7])
    #bc_f7.apply(fe.PETScVector(sysMatLumped[7]))
    bc_f7.apply(streamer.advectionMats[7])
    bc_f7.apply(streamer.doubleAdvectionMats[7])
    
    bc_f4.apply(streamer.sysMatStream[4])
    #bc_f4.apply(fe.PETScVector(sysMatLumped[4]))
    bc_f4.apply(streamer.advectionMats[4])
    bc_f4.apply(streamer.doubleAdvectionMats[4])
    
    bc_f8.apply(streamer.sysMatStream[8])
    #bc_f8.apply(fe.PETScVector(sysMatLumped[8]))
    bc_f8.apply(streamer.advectionMats[8])
    bc_f8.apply(streamer.doubleAdvectionMats[8])
    
    
    streamingPrevTimeVecs= [simState.f_star[0].vector().copy() for _ in range(Q)]
    advectionVecs = [simState.f_star[0].vector().copy() for _ in range(Q)]
    doubleAdvectionVecs =[simState.f_star[0].vector().copy() for _ in range(Q)]
    #rhsVecStreaming = [simState.f_star[0].vector().copy() for _ in range(Q)]
    
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
        
        f_vals = np.array([simState.f_n[idx].vector().get_local() for idx in range(Q)])
        
        forceVals_x = forceDensity_x.vector().get_local()
        #forceVals_x = forceVals_x.reshape((-1, mesh.geometry().dim()))
        
        forceVals_y = forceDensity_y.vector().get_local()
        #forceVals_y = forceVals_y.reshape((-1, mesh.geometry().dim()))
    
        # Compute rho and u as numpy arrays over all DOFs
        rho = f_vals.sum(axis=0)                          # shape (n_dofs,)
        ux  = (xi_arr[:,0,None] * f_vals).sum(axis=0) / rho + forceVals_x*dt/(2*rho)
        uy  = (xi_arr[:,1,None] * f_vals).sum(axis=0) / rho + forceVals_y*dt/(2*rho)
        vel = np.stack([ux, uy])
        cu = xi_arr[:,0,None]*ux + xi_arr[:,1,None]*uy        # (9, n_dofs)
        u2 = ux**2 + uy**2                                    # (n_dofs,)
        feq = w[:,None] * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)
        
        # Now for the foce term
        u_dot_F = ux * forceVals_x + uy * forceVals_y   # (n_dofs,)
        ck_dot_F = xi_arr @ np.column_stack((forceVals_x,forceVals_y)).T   # shape (Q, n_dofs)
        
        ck_dot_u = xi_arr[:,0,None]*ux + xi_arr[:,1,None]*uy   # (9, n_dofs) -- same as your 'cu'
    
        force_term = w[:, None] * (
              ck_dot_F / c_s**2
            + (ck_dot_u * ck_dot_F) / c_s**4   # ← ck_dot_u, not u_dot_F
            - u_dot_F[None, :] / c_s**2        # ← minus sign
        )
        
    
        f_star_np = f_vals - dt/tau*(f_vals - feq) + dt*force_term
        [simState.f_star[idx].vector().set_local(f_star_np[idx,:]) for idx in range(Q)]
        rho = f_star_np.sum(axis=0)
        ux  = (xi_arr[:,0,None] * f_vals).sum(axis=0) / rho + forceVals_x*dt/(2*rho)
        uy  = (xi_arr[:,1,None] * f_vals).sum(axis=0) / rho + forceVals_y*dt/(2*rho)
    
        #print("collision_time =", post_coll_time - pre_coll_time)
        
    
        pre_stream_time = time.time()
        # Assemble RHS vectors for streaming step
        
        streamer.assembleRhsLumping(simState.f_star, dt)
        #print("stream assemble =", post_assemble_stream_time - pre_stream_time)
        
        
    
    
        pre_assign_time = time.time()
        f5_lower_func.assign(simState.f_star[7] )
        f2_lower_func.assign(simState.f_star[4] )
        f6_lower_func.assign(simState.f_star[8] )
        f7_upper_func.assign(simState.f_star[5] )
        f4_upper_func.assign(simState.f_star[2] )
        f8_upper_func.assign(simState.f_star[6] )
        post_assign_time = time.time()
        #print("assign time = ", post_assign_time - pre_assign_time)
    
        pre_apply_time = time.time()
        # Apply BCs for distribution functions 5, 2, and 6
        bc_f5.apply(streamer.rhsVecStreaming[5])
        bc_f2.apply(streamer.rhsVecStreaming[2])
        bc_f6.apply(streamer.rhsVecStreaming[6])
    
        # Apply BCs for distribution functions 7, 4, 8
        bc_f7.apply(streamer.rhsVecStreaming[7])
        bc_f4.apply(streamer.rhsVecStreaming[4])
        bc_f8.apply(streamer.rhsVecStreaming[8])
        post_apply_time = time.time()
        #print("time to apply BCs ", post_apply_time - pre_apply_time)
    
        pre_stream_time = time.time()
        # Solve linear system for streaming step
        
        simState.f_nP1 = streamer.solveSysLumping(simState.f_nP1)
        
        # for idx in range(Q):
        #     #solver_list[idx].solve(simState.f_nP1[idx].vector(), rhsVecStreaming[idx])
        #     vi = fe.as_backend_type(streamer.rhsVecStreaming[idx]).vec()
        #     simState.f_nP1[idx].vector().vec().pointwiseDivide(
        #         vi,
        #         streamer.sysMatLumped[idx])
       
        bc_f5.apply(simState.f_nP1[5].vector())
        bc_f2.apply(simState.f_nP1[2].vector())
        bc_f6.apply(simState.f_nP1[6].vector())
    
        # Apply BCs for distribution functions 7, 4, 8
        bc_f7.apply(simState.f_nP1[7].vector())
        bc_f4.apply(simState.f_nP1[4].vector())
        bc_f8.apply(simState.f_nP1[8].vector())
        post_stream_time = time.time()
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
