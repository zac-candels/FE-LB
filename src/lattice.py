import numpy as np
import fenics as fe


class D2Q9:

    def __init__(self):

        self.Q = 9

        self.xi = [
            fe.Constant((0,0)),
            fe.Constant((1,0)),
            fe.Constant((0,1)),
            fe.Constant((-1,0)),
            fe.Constant((0,-1)),
            fe.Constant((1,1)),
            fe.Constant((-1,1)),
            fe.Constant((-1,-1)),
            fe.Constant((1,-1))
        ]

        self.weights = np.array(
            [
             4/9,
             1/9,1/9,1/9,1/9,
             1/36,1/36,1/36,1/36
            ]
        )
        
        self.xi_arr = np.array(
            [[0,0],[1,0],[0,1],[-1,0],[0,-1],
                               [1,1],[-1,1],[-1,-1],[1,-1]], dtype=float
            )
            
            