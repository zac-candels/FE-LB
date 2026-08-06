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
import allenCahnModule
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
    # Lattice speed of sound
    c_s = params["c_s"]
    # Number of discrete velocities
    Q = params["Q"]
    # Initial density
    tau = params["tau"]
    R0 = params["R0"]
    xc = params["xc"]
    yc = params["yc"]
    theta_deg = params["theta_deg"]
    A = params["A"]
    kappa = params["kappa"]
    M_tilde = params["M_tilde"]
    lumping = params["lumping"]
    forceInCollisionStreaming = params["forceInCollisionStreaming"]
    # D2Q9 lattice velocities
    
    initDropDiam = 2*R0 
    interfaceThickness = np.sqrt(kappa/A)
    theta = theta_deg * np.pi / 180
    
    latticeClass = lattice.D2Q9()
    xi = latticeClass.xi
    xi_arr = latticeClass.xi_arr
    
    mesh, V, Vvec = meshAndFnSpaces.create_mesh(dim, L_x, L_y, nx, ny)
    
    
    h = mesh.hmin()
    dt = 0.005*h/np.sqrt(2)
    beta_mass_diff = 0.01*dt
    num_steps = int(np.ceil(T/dt))
    
    outDirName = writeData.create_output_directory(dt, h, name="forceModule")
    
    
    simState = finiteElementFunctions.SimulationState(V, Vvec, Q)
    
    forceDensity_x = fe.Function(V)
    forceDensity_y = fe.Function(V)
    
    ac = allenCahnModule.allenCahn(V, xc, yc, initDropDiam, interfaceThickness,
                 M_tilde, beta_mass_diff, A, kappa,
                 theta)
    
    forceDensity = ac.phi_n*fe.grad(ac.mu_n)
    
    simState.f_n= initialize.initializeDistributionsMultiPhase(simState.f_n,
                                                               forceDensity,
                                                               V,
                                                               latticeClass,
                                                               c_s, dt, tau)
    
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
    
    
    #streamer.createLinearformsForceInStreaming(simState,
    #                                           dt,
    #                                           forceDensityTuple)
    
    # Create MeshFunction for boundary markers
    boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)

    # Subdomain for bottom wall
    class Bottom(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fe.near(x[1], 0.0)

    bottom = Bottom()
    bottom.mark(boundaries, 1)   # assign ID = 1 to bottom boundary
    ds_bottom = fe.Measure("ds", domain=mesh, subdomain_data=boundaries, subdomain_id=1)
    
    

    

    
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
        
        ac.assembleRhsPhiLumping()
        
        forceVals_x, forceVals_y = collision.computeForce(forceDensity,
                                                            simState.v,
                                                            forceVec_x,
                                                            forceVec_y,
                                                            forceDensity_x,
                                                            forceDensity_y,
                                                            streamer.massMat)
        
    
        simState.f_star = collision.collideLocalForceInStreaming(simState.f_n, 
                                       simState.f_star, 
                                       latticeClass,
                                       (forceVals_x, forceVals_y),
                                       tau,
                                       dt)
    
        #print("collision_time =", post_coll_time - pre_coll_time)
        
    
        pre_stream_time = time.time()
        # Assemble RHS vectors for streaming step
        
        streamer.assembleRhsLumping(simState.f_star, dt, Force_density)
        #print("stream assemble =", post_assemble_stream_time - pre_stream_time)
    
        lower_bcs.update(simState.f_star)
        upper_bcs.update(simState.f_star)
    
        lower_bcs.applyRhsVec(streamer.rhsVecStreaming)
        upper_bcs.applyRhsVec(streamer.rhsVecStreaming)

        # Solve linear system for streaming step
        
        simState.f_nP1 = streamer.solveSysLumping(simState.f_nP1)
       
        lower_bcs.applyF_nP1(simState.f_nP1)
        upper_bcs.applyF_nP1(simState.f_nP1)
        
        #print("time to solve stream sys ", post_stream_time - pre_stream_time, "\n\n\n\n")
    
        # Update previous solutions
    
        for idx in range(Q):
            simState.f_n[idx].assign(simState.f_nP1[idx])
            

    
        if n % 5000 == 0:
            testOutput.writeOutput(n, xi,
                                                simState.f_n,
                                                V,
                                                Vvec,
                                                simState.vel_n,
                                                u_max,
                                                L_x,
                                                L_y,
                                                tau,
                                                dt,
                                                Force_density,
                                                outDirName )

    


main()
