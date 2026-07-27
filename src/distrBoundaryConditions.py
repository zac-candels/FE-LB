import fenics as fe 

tol = 1e-4
class BounceBackBoundary:
    
    def __init__(self, V, streamer, f_list, boundary_predicate, index_pairs):
        self.V = V
        self.boundary_predicate = boundary_predicate
        self.index_pairs = index_pairs
        self.streamer = streamer
        self.f_list = f_list 
        self.BC_fn_list = [fe.Function(V) for i in range(6)]
        self.bcs = None
    
        self._create_bcs()
        
    def _create_bcs(self):

        self.bcs = {}
        for wall_idx, conj_idx in self.index_pairs:
            func = fe.Function(self.V)
            fe.project(self.f_list[conj_idx], self.V, function=func)
            bc = fe.DirichletBC(self.V, func, self.boundary_predicate)
            bc.apply(self.streamer.sysMatStream[wall_idx])
            bc.apply(self.streamer.advectionMats[wall_idx])
            bc.apply(self.streamer.doubleAdvectionMats[wall_idx])
            self.bcs[wall_idx] = (bc, func)
            
    def update(self, f_star):
        
        for wall_idx, conj_idx in self.index_pairs:
            self.bcs[wall_idx][1].assign(f_star[conj_idx])
            
    def applyRhsVec(self, Vec):
        
        for wall_idx, conj_idx in self.index_pairs:
            
            self.bcs[wall_idx][0].apply(Vec[wall_idx])
            
    def applyF_nP1(self, Vec):
        
        for wall_idx, conj_idx in self.index_pairs:
            
            self.bcs[wall_idx][0].apply(Vec[wall_idx].vector())

    