import sys
sys.path.insert(0, "/home/zcandels/refactor/src")
import lattice
import meshAndFnSpaces
from postProcessing import writeData
import finiteElementFunctions
import moments
import streamingModule
import collision
import distrBoundaryConditions
import initialize 
import forceModule
from postProcessing import testOutput
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
    # Initial density
    rho_init = params["rho_init"]
    u_max = params["u_max"]
    tau = params["tau"]
    lumping = params["lumping"]
    forceInCollisionStreaming = params["forceInCollisionStreaming"]
    # D2Q9 lattice velocities
    
    latticeClass = lattice.D2Q9()
    xi = latticeClass.xi
    xi_arr = latticeClass.xi_arr
    
    mesh, V, Vvec = meshAndFnSpaces.create_mesh(dim, L_x, L_y, nx, ny)


    h = mesh.hmin()
    dt = 0.005*h/np.sqrt(2)
    num_steps = int(np.ceil(T/dt))

    outDirName = writeData.create_output_directory(dt, h, name="forceModule")
    
 
    simState = finiteElementFunctions.SimulationState(V, Vvec, Q)
    
    forceDensity_x = fe.Function(V)
    forceDensity_y = fe.Function(V)
    
        
    simState.f_n= initialize.initializeDistributions(simState.f_n,
                                                               Force_density,
                                                               V,
                                                               latticeClass,
                                                               c_s, dt)
    
    
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
    
    lower_pairs = [(2,4), (5,7), (6,8)]
    upper_pairs = [(4,2), (7,5), (8, 6)]
    upper_bcs = distrBoundaryConditions.BounceBackBoundary(V,
                                                           streamer,
                                                           simState.f_n,
                                                           Bdy_Upper,
                                                           upper_pairs)
    lower_bcs = distrBoundaryConditions.BounceBackBoundary(V,
                                                           streamer,
                                                           simState.f_n,
                                                           Bdy_Lower,
                                                           lower_pairs)
    
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
        
        forceVals_x, forceVals_y = forceModule.computeForce(Force_density,
                                                            simState.v,
                                                            forceVec_x,
                                                            forceVec_y,
                                                            forceDensity_x,
                                                            forceDensity_y,
                                                            streamer.massMat)
    
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
            testOutput.writeOutput(n, 
                                                simState.f_n,
                                                V,
                                                Vvec,
                                                simState.vel_n,
                                                u_max,
                                                L_x,
                                                L_y,
                                                tau,
                                                dt,
                                                forceDensityTuple,
                                                outDirName )
            

main()
