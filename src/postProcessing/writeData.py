import sys
sys.path.insert(0, "/home/zcandels/refactor")
from pathlib import Path
import shutil
import fenics as fe 
import moments 
import os
import numpy as np
import matplotlib.pyplot as plt



def create_output_directory(dt, h, name="outputDir"):
    
    workdir = Path.cwd()

    out_dir = workdir / f"{name}_dt{dt:.2e}_h{h:.2f}"

    if out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir


def saveDataToFile_poiseuille(n, f_n, V, Vvec, vel_n, u_max, L_x, L_y,
                              tau, dt, forceDensityTuple, outDirName ):

    
        xi_arr = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],
                           [1,1],[-1,1],[-1,-1],[1,-1]], dtype=float)

    
        Q = len(xi_arr)
        print("n = ", n)
        vel_expr = moments.getVel(f_n, xi_arr, forceDensityTuple, dt)
        fe.project(vel_expr, Vvec, function=vel_n)
        #vel_file.write(vel_n, t)
        
        # print("max |drho|   =", np.max(np.abs(rho_diff)), flush=True)
        # print("max |d_momentum_x|=", np.max(np.abs(momx_diff)), flush=True)
        # print("max |d_momentum_y|=", np.max(np.abs(momy_diff)), flush=True)

        # log_file.write(f"{np.max(np.abs(rho_diff)):15.4f} {np.max(np.abs(momx_diff)):15.4f}  {np.max(np.abs(momy_diff)):15.4f} \n")
        # log_file.flush()
        
        u_new, v_new = 0, 0
        
        for i in range(Q):
            xi_new = xi_arr[i].values()
            u_new += f_n[i].vector().get_local()*xi_new[0]
            v_new += f_n[i].vector().get_local()*xi_new[1]

        u_e = fe.Expression('u_max*( 1 - pow( (2*x[1]/L_y -1), 2 ) )',
                            degree=2, u_max=u_max, L_y=L_y)
        u_e = fe.interpolate(u_e, V)
        error = np.linalg.norm(u_e.vector().get_local() - u_new)
        print('max u:', u_new.max(), flush=True)

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
            u_at_point = vel_n(point)
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