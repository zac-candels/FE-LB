import sys
sys.path.insert(0, "/home/zcandels/refactor/src")
import lattice
import moments
import meshAndFnSpaces
import initialize
import streamingModule
from postProcessing import writeData
import finiteElementFunctions
import distrBoundaryConditions
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

start_time = time.time()

comm = fe.MPI.comm_world
rank = fe.MPI.rank(comm)

plt.close('all')





def main():
    with open("params.json") as file:
        params = json.load(file)

    dim = params["dim"]
    T = params["Tfinal"]
    Re = params["Re"]
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
    
    mesh, V, Vvec = meshAndFnSpaces.create_mesh(dim, L_x, L_y, nx, ny)


    h = mesh.hmin()
    dt = 0.00025*h/np.sqrt(2)
    num_steps = int(np.ceil(T/dt))


    outDirName = writeData.create_output_directory(dt, h, name="outputDir")

    simState = finiteElementFunctions.SimulationState(V, Vvec, Q)

    forceDensity_n = fe.interpolate(Force_density, Vvec)


    forceDensity_x = fe.Function(V)
    forceDensity_y = fe.Function(V)

    # Define velocity



    # # Initialize distribution functions. We will use
    # f_i^{0} \gets f_i^{0, eq}( \rho_0, \bar{u}_0 ),
    # where \bar{u}_0 = u_0 - F\Delta t/( 2 \rho_0 ).
    # Here we will take u_0 = 0.
    for idx in range(Q):
        simState.f_n[idx] = (fe.project(initialize.f_equil_init(
            idx, Force_density, dt, xi, w, c_s), V))


    bouncebackBCs = distrBoundaryConditions.BounceBackBoundary( V,
                                                               simState.f_n,
                                                               L_y,
                                                               tol=1e-8)
    
    bouncebackBCs._create_bcs(L_y, tol=1e-8)

    streaming = streamingModule.StreamingOperator(V, simState,
                                                  latticeClass, dt)

    sysMatStream = streaming.sysMatStream
    advectionMats = streaming.advectionMats
    doubleAdvectionMats = streaming.doubleAdvectionMats

    vel_file = fe.XDMFFile(comm, f"{outDirName}/vel.xdmf")
    vel_file.parameters["flush_output"] = True
    vel_file.parameters["functions_share_mesh"] = True
    vel_file.parameters["rewrite_function_mesh"] = False


    # Apply BCs to matrices for distribution functions 5, 2, and 6
    bouncebackBCs.lower["f5"].apply(sysMatStream[5])
    #bc_f5.apply(fe.PETScVector(sysMatLumped[5]))
    bouncebackBCs.lower["f5"].apply(advectionMats[5])
    bouncebackBCs.lower["f5"].apply(doubleAdvectionMats[5])

    bouncebackBCs.lower["f2"].apply(sysMatStream[2])
    #bc_f2.apply(fe.PETScVector(sysMatLumped[2]))
    bouncebackBCs.lower["f2"].apply(advectionMats[2])
    bouncebackBCs.lower["f2"].apply(doubleAdvectionMats[2])

    bouncebackBCs.lower["f6"].apply(sysMatStream[6])
    #bc_f6.apply(fe.PETScVector(sysMatLumped[6]))
    bouncebackBCs.lower["f6"].apply(advectionMats[6])
    bouncebackBCs.lower["f6"].apply(doubleAdvectionMats[6])

    # Apply BCs to matrices for distribution functions 7, 4, 8
    bouncebackBCs.upper["f7"].apply(sysMatStream[7])
    #bc_f7.apply(fe.PETScVector(sysMatLumped[7]))
    bouncebackBCs.upper["f7"].apply(advectionMats[7])
    bouncebackBCs.upper["f7"].apply(doubleAdvectionMats[7])

    bouncebackBCs.upper["f4"].apply(sysMatStream[4])
    #bc_f4.apply(fe.PETScVector(sysMatLumped[4]))
    bouncebackBCs.upper["f4"].apply(advectionMats[4])
    bouncebackBCs.upper["f4"].apply(doubleAdvectionMats[4])

    bouncebackBCs.upper["f8"].apply(sysMatStream[8])
    #bc_f8.apply(fe.PETScVector(sysMatLumped[8]))
    bouncebackBCs.upper["f8"].apply(advectionMats[8])
    bouncebackBCs.upper["f8"].apply(doubleAdvectionMats[8])


    streamingPrevTimeVecs= [simState.f_star_coll[0].vector().copy() for _ in range(Q)]
    advectionVecs = [simState.f_star_coll[0].vector().copy() for _ in range(Q)]
    doubleAdvectionVecs =[simState.f_star_coll[0].vector().copy() for _ in range(Q)]
    rhsVecStreaming = [simState.f_star_coll[0].vector().copy() for _ in range(Q)]

    forceVec_x = simState.f_star_coll[0].vector().copy()
    forceVec_y = simState.f_star_coll[0].vector().copy()

    A_blocks = []

    for i in range(Q):

        M = fe.as_backend_type(streaming.M_lumped).mat()

        K = fe.as_backend_type(advectionMats[i]).mat()
        D = fe.as_backend_type(doubleAdvectionMats[i]).mat()

        # IMPORTANT: do NOT use copy() unless necessary
        A_i = M.copy()
        A_i.axpy(-dt, K)
        A_i.axpy(0.5*dt**2, D)

        A_blocks.append([
            A_i if j == i else None for j in range(Q)
        ])

    blockStreamingAssemblyMatrix = PETSc.Mat().createNest(A_blocks)
    blockStreamingAssemblyMatrix.assemble()
        
    rhsVecsStreaming = [fe.as_backend_type(rhsVecStreaming[i]).vec() for i in range(Q)]
    blockRhsVecsStreaming = PETSc.Vec().createNest(rhsVecsStreaming)
    f_starVec = simState.f_star_coll[0].vector().copy()
    FstarVecs = [fe.as_backend_type(f_starVec).vec() for i in range(Q)]
    blockFstarVecs = PETSc.Vec().createNest(FstarVecs)

    xi_arr = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],
                       [1,1],[-1,1],[-1,-1],[1,-1]], dtype=float)


        
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
        
        fe.solve(streaming.mass_mat, forceDensity_x.vector(), forceVec_x)
        # petscForce_x = fe.as_backend_type(forceVec_x)
        # forceDensity_x.vector().vec().pointwiseDivide(petscForce_x.vec(), M_petsc)
        fe.solve(streaming.mass_mat, forceDensity_y.vector(), forceVec_y)
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
        [simState.f_star_coll[idx].vector().set_local(f_star_np[idx,:]) for idx in range(Q)]

        #print("collision_time =", post_coll_time - pre_coll_time)
        

        pre_stream_time = time.time()
        # Assemble RHS vectors for streaming step
        for idx in range(Q):
            streaming.M_lumped.mult(
                simState.f_star_coll[idx].vector(),
                streamingPrevTimeVecs[idx])
            advectionMats[idx].mult(simState.f_star_coll[idx].vector(),
                                    advectionVecs[idx])
            doubleAdvectionMats[idx].mult(simState.f_star_coll[idx].vector(),
                                          doubleAdvectionVecs[idx])

            rhsVecStreaming[idx].zero()
            rhsVecStreaming[idx].axpy(1.0, streamingPrevTimeVecs[idx])
            rhsVecStreaming[idx].axpy(-dt, advectionVecs[idx])
            rhsVecStreaming[idx].axpy(0.5*dt**2, doubleAdvectionVecs[idx])
        #print("stream assemble =", post_assemble_stream_time - pre_stream_time)
        
        


        bouncebackBCs.update(simState.f_star_coll)
        post_assign_time = time.time()
        #print("assign time = ", post_assign_time - pre_assign_time)

        pre_apply_time = time.time()
        # Apply BCs for distribution functions 5, 2, and 6
        bouncebackBCs.lower["f5"].apply(rhsVecStreaming[5])
        bouncebackBCs.lower["f2"].apply(rhsVecStreaming[2])
        bouncebackBCs.lower["f6"].apply(rhsVecStreaming[6])
        
        bouncebackBCs.upper["f7"].apply(rhsVecStreaming[7])
        bouncebackBCs.upper["f4"].apply(rhsVecStreaming[4])
        bouncebackBCs.upper["f8"].apply(rhsVecStreaming[8])
        post_apply_time = time.time()
        #print("time to apply BCs ", post_apply_time - pre_apply_time)

        subvecs = blockRhsVecsStreaming.getNestSubVecs()
        pre_stream_time = time.time()
        # Solve linear system for streaming step
        for idx in range(Q):
            #solver_list[idx].solve(f_nP1[idx].vector(), rhsVecStreaming[idx])
            vi = fe.as_backend_type(rhsVecStreaming[idx]).vec()
            simState.f_star_stream[idx].vector().vec().pointwiseDivide(
                vi, 
                streaming.sysMatLumped[idx])
       
        bouncebackBCs.lower["f5"].apply(simState.f_star_stream[5].vector())
        bouncebackBCs.lower["f2"].apply(simState.f_star_stream[2].vector())
        bouncebackBCs.lower["f6"].apply(simState.f_star_stream[6].vector())
        
        bouncebackBCs.upper["f7"].apply(simState.f_star_stream[7].vector())
        bouncebackBCs.upper["f4"].apply(simState.f_star_stream[4].vector())
        bouncebackBCs.upper["f8"].apply(simState.f_star_stream[8].vector())
        post_stream_time = time.time()
        #print("time to solve stream sys ", post_stream_time - pre_stream_time, "\n\n\n\n")
        
        
        f_vals = np.array([simState.f_star_stream[idx].vector().get_local() for idx in range(Q)])
        
        forceVals_x = forceDensity_x.vector().get_local()
        #forceVals_x = forceVals_x.reshape((-1, mesh.geometry().dim()))
        
        forceVals_y = forceDensity_y.vector().get_local()
        #forceVals_y = forceVals_y.reshape((-1, mesh.geometry().dim()))

        # Compute rho and u as numpy arrays over all DOFs
        rho = f_vals.sum(axis=0)                          # shape (n_dofs,)
        ux  = (xi_arr[:,0,None] * f_vals).sum(axis=0) / rho + forceVals_x*(dt/2)/(2*rho)
        uy  = (xi_arr[:,1,None] * f_vals).sum(axis=0) / rho + forceVals_y*(dt/2)/(2*rho)
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
        

        f_new = f_vals - dt/tau*(f_vals - feq) + dt*force_term
        [simState.f_nP1[idx].vector().set_local(f_new[idx,:]) for idx in range(Q)]


        # Update previous solutions

        for idx in range(Q):
            simState.f_n[idx].assign(simState.f_nP1[idx])
            
        a=1
            
        #fe.project(getVel(f_n), Vvec, function=vel_n)
        #fe.project(getDens(f_n), V, function=rho_n)

        if n % 5000 == 0:
            
            print("n = ", n)
            vel_expr = moments.getVel(simState.f_n, xi, forceDensityTuple, dt)
            fe.project(vel_expr, Vvec, function=simState.vel_n)
            #vel_file.write(vel_n, t)
            
            # print("max |drho|   =", np.max(np.abs(rho_diff)), flush=True)
            # print("max |d_momentum_x|=", np.max(np.abs(momx_diff)), flush=True)
            # print("max |d_momentum_y|=", np.max(np.abs(momy_diff)), flush=True)

            # log_file.write(f"{np.max(np.abs(rho_diff)):15.4f} {np.max(np.abs(momx_diff)):15.4f}  {np.max(np.abs(momy_diff)):15.4f} \n")
            # log_file.flush()
            
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
            u_max = 0.1
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