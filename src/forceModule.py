import fenics as fe

def computeForce(Force_density, v, forceVec_x, forceVec_y,
                 forceDensity_x, forceDensity_y, massMat):
    
    fe.assemble(Force_density.values()[0]*v*fe.dx, tensor=forceVec_x )
    fe.assemble(v*fe.dx, tensor=forceVec_y)
    forceVec_y.vec().scale(0)
    
    fe.solve(massMat, forceDensity_x.vector(), forceVec_x)
    # petscForce_x = fe.as_backend_type(forceVec_x)
    # forceDensity_x.vector().vec().pointwiseDivide(petscForce_x.vec(), M_petsc)
    fe.solve(massMat, forceDensity_y.vector(), forceVec_y)
    # petscForce_y = fe.as_backend_type(forceVec_y)
    # forceDensity_y.vector().vec().pointwiseDivide(petscForce_y.vec(), M_petsc)
    #print("project force time = ", projectForceTimeEnd - projectForceTimeStart)
    
    # We will try to do collision locally, since it is a pure
    # time-dependnet ODE
    
    forceVals_x = forceDensity_x.vector().get_local()
    #forceVals_x = forceVals_x.reshape((-1, mesh.geometry().dim()))
    
    forceVals_y = forceDensity_y.vector().get_local()
    #forceVals_y = forceVals_y.reshape((-1, mesh.geometry().dim()))
    
    return forceVals_x, forceVals_y