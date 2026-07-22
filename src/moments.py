import fenics as fe

def getDens(f_list):

    rho = f_list[0]

    for f in f_list[1:]:
        rho += f

    return rho


def getVel(f_list, xi_arr, forceDensityTuple, dt):

    momentum = f_list[0]*xi_arr[0]

    for i in range(1, len(f_list)):
        momentum += f_list[i]*xi_arr[i]

    rho = getDens(f_list)

    u_raw = momentum/rho

    force_correction = fe.Constant(forceDensityTuple )* dt / (2*rho)
    

    return u_raw + force_correction