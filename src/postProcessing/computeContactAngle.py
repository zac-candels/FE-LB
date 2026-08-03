import numpy as np
import fenics as fe
from scipy import optimize 


def calc_R(xc, yc, x_coords, y_coords):
    #calculate the distance of each 2D points from the center (xc, yc) 
    return np.sqrt((x_coords-xc)**2 + (y_coords-yc)**2)

def f_2(c):
    #calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) 
    Ri = calc_R(*c)
    return Ri - Ri.mean()

def computeContactAngle_gradPhi(c_n, h, Cn, mesh, comm, rank):
   
    if rank == 0:
        print("in compute angle fn", flush=True)
    
    coords = mesh.coordinates()

    x_min_local = np.min(coords[:, 0])
    x_max_local = np.max(coords[:, 0])
    
    x_min = fe.MPI.min(comm, x_min_local)
    x_max = fe.MPI.max(comm, x_max_local)
    
    L_x = x_max - x_min
    
    x0 = x_min + 0.5*L_x

    tol = h
    
    Vvec = fe.VectorFunctionSpace(mesh, "DG", 0)
    grad_c_fn = fe.project(fe.grad(c_n), Vvec, solver_type="cg",
                           preconditioner_type="jacobi")
    angles = []
    n_vec = np.array([0.0, 0.0, -1.0])
    
    barycenters = []
    barycenter_vals = []
    
    if rank == 0:
        print("about to start creating nodal dictionary in compute angle fn.", flush=True)
    for cell in fe.cells(mesh):
        
        midpt = cell.midpoint().array()
        
        if abs(midpt[0] - x0) > tol:
            continue
        
        midpt = tuple( (midpt[0], midpt[1], midpt[2]) )
        barycenters.append( midpt )
        barycenter_vals.append( c_n(midpt) )
    
    # Build dictionary
    nodal_dict = {
    tuple(coord): val
    for coord, val in zip(barycenters, barycenter_vals)
    }

    if rank == 0:
        print("in compute angle function, created nodal dictionary", flush=True)
    
    # Filter by z-coordinate
    nodal_dict = {
        coord: value
        for coord, value in nodal_dict.items() 
        if coord[2] < 1.5*h}
    
    if rank == 0:
        print("in compute angle function, filtered nodal dictionary for small z", flush=True)
    # Filter by order parameter value
    nodal_dict = {
        coord: value
        for coord, value in nodal_dict.items() 
        if -0.15 < value < 0.15}
    
    if rank == 0:
        print("in compute function, filtered nodal dictionary for phi", flush=True)
    # Determine left-most interfacial point
    if len(nodal_dict) > 0:
        local_min_y = min(coord[1] for coord in nodal_dict.keys())
    else:
        local_min_y = np.inf
    
    min_y = fe.MPI.min(comm, local_min_y)

    # Filter points so we get rid of points near right CL
    nodal_dict = {
        coord: value
        for coord, value in nodal_dict.items() 
        if coord[1] > min_y + 5*Cn}
    
    iter = 0
    for coord, value in nodal_dict.items():
        iter += 1
        #print("coord is", coord)
        grad_c = np.array(grad_c_fn(coord))
        cos_theta = np.dot(grad_c, n_vec) / np.linalg.norm(grad_c)
        angles.append( np.arccos(cos_theta))

    #print("Averaged over ", iter, " points")
        
    local_sum = np.sum(angles)
    local_n = len(angles)
    
    global_sum = fe.MPI.sum(comm, local_sum)
    global_n = fe.MPI.sum(comm, local_n)
    
    theta_avg = global_sum / global_n
    theta_avg = theta_avg * 180 / np.pi
    
    return theta_avg




def computeContactAngle_gradPhi(c_n, h, Cn, mesh, comm, rank):

    if rank == 0:
        print("in compute angle fn", flush=True)

    coords = mesh.coordinates()

    x_min_local = np.min(coords[:, 0])
    x_max_local = np.max(coords[:, 0])

    x_min = fe.MPI.min(comm, x_min_local)
    x_max = fe.MPI.max(comm, x_max_local)

    L_x = x_max - x_min
    x0 = x_min + 0.5 * L_x

    tol = h

    Vvec = fe.VectorFunctionSpace(mesh, "DG", 0)
    grad_c_fn = fe.project(fe.grad(c_n), Vvec, solver_type="cg",
                           preconditioner_type="jacobi")
    angles = []
    n_vec = np.array([0.0, 0.0, -1.0])

    barycenters = []
    barycenter_vals = []

    # Safe, ghost-aware per-vertex values for c_n (P1), indexed by local vertex id.
    c_vertex_vals = c_n.compute_vertex_values(mesh)

    tdim = mesh.topology().dim()
    ghost_offset = mesh.topology().ghost_offset(tdim)

    # DG0 grad dofs, local vector (safe: only touched for owned cells below)
    Vvec_dofmap = Vvec.dofmap()
    grad_vals_local = grad_c_fn.vector().get_local()

    if rank == 0:
        print("about to start creating nodal dictionary in compute angle fn.", flush=True)

    midpt_to_cell = {}

    for cell in fe.cells(mesh):

        # Skip ghost cells: only process cells this rank actually owns,
        # otherwise points near partition boundaries get double-counted
        # across ranks and ghost vertex dofs can be out of range.
        if cell.index() >= ghost_offset:
            continue

        midpt = cell.midpoint().array()

        if abs(midpt[0] - x0) > tol:
            continue

        vertex_ids = cell.entities(0)
        c_val = c_vertex_vals[vertex_ids].mean()

        midpt = tuple((midpt[0], midpt[1], midpt[2]))
        barycenters.append(midpt)
        barycenter_vals.append(c_val)
        midpt_to_cell[midpt] = cell.index()

    # Build dictionary
    nodal_dict = {
        tuple(coord): val
        for coord, val in zip(barycenters, barycenter_vals)
    }

    if rank == 0:
        print("in compute angle function, created nodal dictionary", flush=True)

    # Filter by z-coordinate
    nodal_dict = {
        coord: value
        for coord, value in nodal_dict.items()
        if coord[2] < 1.5 * h}

    if rank == 0:
        print("in compute angle function, filtered nodal dictionary for small z", flush=True)

    # Filter by order parameter value
    nodal_dict = {
        coord: value
        for coord, value in nodal_dict.items()
        if -0.15 < value < 0.15}

    if rank == 0:
        print("in compute function, filtered nodal dictionary for phi", flush=True)

    # Determine left-most interfacial point
    if len(nodal_dict) > 0:
        local_min_y = min(coord[1] for coord in nodal_dict.keys())
    else:
        local_min_y = np.inf

    min_y = fe.MPI.min(comm, local_min_y)

    # Filter points so we get rid of points near right CL
    nodal_dict = {
        coord: value
        for coord, value in nodal_dict.items()
        if coord[1] > min_y + 5 * Cn}

    iter = 0
    for coord, value in nodal_dict.items():
        iter += 1

        cell_index = midpt_to_cell.get(coord, None)
        if cell_index is None:
            continue

        cell_dofs = Vvec_dofmap.cell_dofs(cell_index)  # gdim dofs, one per component
        grad_c = grad_vals_local[cell_dofs]

        cos_theta = np.dot(grad_c, n_vec) / np.linalg.norm(grad_c)
        angles.append(np.arccos(cos_theta))

    local_sum = np.sum(angles)
    local_n = len(angles)

    global_sum = fe.MPI.sum(comm, local_sum)
    global_n = fe.MPI.sum(comm, local_n)

    theta_avg = global_sum / global_n
    theta_avg = theta_avg * 180 / np.pi

    return theta_avg



def computeContactAngle_heightDiam(phi_n, h, Cn, mesh, comm, rank):

    coords = mesh.coordinates()
    x_min_local = np.min(coords[:, 0])
    x_max_local = np.max(coords[:, 0])

    x_min = fe.MPI.min(comm, x_min_local)
    x_max = fe.MPI.max(comm, x_max_local)

    L_x = x_max - x_min

    x0 = x_min + 0.5 * L_x
    tol = h

    barycenters = []
    barycenter_vals = []

    # Safe, ghost-aware per-vertex values for phi_n (P1), indexed by local vertex id.
    phi_vertex_vals = phi_n.compute_vertex_values(mesh)

    tdim = mesh.topology().dim()
    ghost_offset = mesh.topology().ghost_offset(tdim)

    for cell in fe.cells(mesh):

        # Skip ghost cells: only process cells this rank actually owns,
        # otherwise points near partition boundaries get double-counted
        # across ranks and ghost vertex indices can be out of range.
        if cell.index() >= ghost_offset:
            continue

        midpt = cell.midpoint().array()

        if abs(midpt[0] - x0) > tol:
            continue

        vertex_ids = cell.entities(0)
        phi_val = phi_vertex_vals[vertex_ids].mean()

        midpt = tuple((midpt[0], midpt[1], midpt[2]))
        barycenters.append(midpt)
        barycenter_vals.append(phi_val)

    # Build dictionary
    nodal_dict = {
        tuple(coord): val
        for coord, val in zip(barycenters, barycenter_vals)
    }

    # Filter by order parameter value
    nodal_dict = {
        coord: value
        for coord, value in nodal_dict.items()
        if -0.15 < value < 0.15}

    # Determine left-most interfacial point
    if len(nodal_dict) > 0:
        local_min_y = min(coord[1] for coord in nodal_dict.keys())
    else:
        local_min_y = np.inf

    min_y = fe.MPI.min(comm, local_min_y)

    # Determine right-most interfacial point
    if len(nodal_dict) > 0:
        local_max_y = max(coord[1] for coord in nodal_dict.keys())
    else:
        local_max_y = -np.inf

    max_y = fe.MPI.max(comm, local_max_y)
    diameter = max_y - min_y

    # Determine height of droplet
    if len(nodal_dict) > 0:
        local_height = max(coord[2] for coord in nodal_dict.keys())
    else:
        local_height = -np.inf

    height = fe.MPI.max(comm, local_height)

    # Compute contact angle in radians
    theta_rad = 2 * np.arctan(2 * height / diameter)
    theta_deg = theta_rad * 180 / np.pi

    return theta_deg

        
def computeContactAngle_regression(c_n, mesh):
    phi_vals = c_n.vector().get_local()
    
    V = c_n.function_space()
    dof_coords = V.tabulate_dof_coordinates()
    
    x_coords = dof_coords[:,0]
    y_coords = dof_coords[:,1]
    
    x_m = np.mean(x_coords)
    y_m = np.mean(y_coords)
    
    center_estimate = x_m, y_m
    
    center_2, ier = optimize.leastsq(f_2, center_estimate)

    x_c, y_c = center_2
    Ri       = calc_R(*center_2)
    Radius       = Ri.mean()
    residu  = sum((Ri - Radius)**2)

    y_min = min(y_coords)

    tol = 1
    hydrophilicity = ""
    if y_min < y_c:
        hydrophilicity = "Hydrophobic"
    elif y_min > y_c:
        hydrophilicity = "Hydrophilic"
    elif abs(y_min - y_c) < tol:
        hydrophilicity = "Neither"
        

    x_min = min(x_coords)
    x_max = max(y_coords)
    # Equation of circle fitted to the extracted droplet interface.
    # 
    x_fit = np.linspace(x_c - Radius**2, x_c + Radius**2, 5000000)
    y_fit_top = y_c + np.sqrt(Radius**2 - (x_fit-x_c)**2)
    y_fit_bottom = y_c + -np.sqrt(Radius**2 - (x_fit-x_c)**2)


    x_fit = x_fit[~np.isnan(y_fit_top)]
    y_fit_top = y_fit_top[~np.isnan(y_fit_top)]
    y_fit_bottom = y_fit_bottom[~np.isnan(y_fit_bottom)]

    circle_top = np.column_stack( [x_fit, y_fit_top] )
    circle_bottom = np.column_stack( [x_fit, y_fit_bottom] )

    circle_top = circle_top[~np.isnan(circle_top).any(axis=1)]
    circle_bottom = circle_bottom[~np.isnan(circle_bottom).any(axis=1)]
    x_fit = circle_bottom[:, 0]

    closest_val_y = np.min(y_coords)

    if hydrophilicity == "Hydrophilic": # ie \theta < 90
        print("\n\n hydrophilic")
        closest_val_x = - np.sqrt( Radius**2 - (closest_val_y - y_c)**2 ) + x_c
        deriv = -(closest_val_x - x_c)/np.sqrt( Radius**2 - (closest_val_x - x_c)**2 )
        CA_1 = 180*np.arctan(deriv)/np.pi
    elif hydrophilicity == "Hydrophobic": # ie \theta > 90
        print("\n\n hydrophobic")
        closest_val_x = - np.sqrt( Radius**2 - (closest_val_y - y_c)**2 ) + x_c
        deriv = (closest_val_x - x_c)/np.sqrt( Radius**2 - (closest_val_x - x_c)**2 )
        CA_1 = 180 + 180*np.arctan(deriv)/np.pi
    elif hydrophilicity == "Neither":
        if abs(Radius**2 - (closest_val_x - x_c)**2) < 2*tol:
            deriv = "undefined"
            CA_1 = 90

        
    print("theta = ", CA_1)
    
    
    
    
    return 