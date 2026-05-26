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

def computeContactAngle_gradPhi(c_n, h, Cn, mesh):
    
    V = c_n.function_space()
    Vvec = fe.VectorFunctionSpace(mesh, "DG", 0)
    grad_c_fn = fe.project(fe.grad(c_n), Vvec)
    angles = []
    n_vec = np.array([0.0, -1.0])
    
    barycenters = []
    barycenter_vals = []
    for cell in fe.cells(mesh):
        
        midpt = cell.midpoint().array()
        midpt = tuple( (midpt[0], midpt[1]) )
        barycenters.append( midpt )
        barycenter_vals.append( c_n(midpt) )
    
    # Build dictionary
    nodal_dict = {
    tuple(coord): val
    for coord, val in zip(barycenters, barycenter_vals)
    }

    
    # Filter by y-coordinate
    nodal_dict = {
        coord: value
        for coord, value in nodal_dict.items() 
        if coord[1] < 1.5*h}
    
    # Filter by order parameter value
    nodal_dict = {
        coord: value
        for coord, value in nodal_dict.items() 
        if -0.3 < value < 0.3}
    
    # Determine left-most interfacial point
    min_x = min(coord[0] for coord in nodal_dict.keys())

    # Filter points so we get rid of points near right CL
    nodal_dict = {
        coord: value
        for coord, value in nodal_dict.items() 
        if coord[0] > min_x + 5*Cn}
    
    iter = 0
    for coord, value in nodal_dict.items():
        iter += 1
        #print("coord is", coord)
        grad_c = np.array(grad_c_fn(coord))
        cos_theta = np.dot(grad_c, n_vec) / np.linalg.norm(grad_c)
        angles.append( np.arccos(cos_theta))

    #print("Averaged over ", iter, " points")
        
    theta_avg = np.mean(angles)
    theta_avg = theta_avg * 180 / np.pi
    
    return theta_avg

def computeContactAngle_heightDiam(phi_n, h, Cn, mesh):

    barycenters = []
    barycenter_vals = []
    for cell in fe.cells(mesh):
        
        midpt = cell.midpoint().array()
        midpt = tuple( (midpt[0], midpt[1]) )
        barycenters.append( midpt )
        barycenter_vals.append( phi_n(midpt) )
    
    # Build dictionary
    nodal_dict = {
    tuple(coord): val
    for coord, val in zip(barycenters, barycenter_vals)
    }

    
    # Filter by order parameter value
    nodal_dict = {
        coord: value
        for coord, value in nodal_dict.items() 
        if -0.3 < value < 0.3}
    
    # Determine left-most interfacial point
    min_x = min(coord[0] for coord in nodal_dict.keys())

    # Determine right-most interfacial point 
    max_x = max(coord[0] for coord in nodal_dict.keys())

    diameter = max_x - min_x 

    # Determine height of droplet 
    height = max(coord[1] for coord in nodal_dict.keys())

    # Compute contact angle in radians
    theta_rad = 2*np.arctan(2*height/diameter)

    theta_deg = theta_rad*180/np.pi 

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