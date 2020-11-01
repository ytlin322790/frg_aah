import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from frg.tool_aah_solver import t_m,v_m,tri,inv_tri,rg_eq_densityvertex,init_vertex
from functools import partial
import multiprocessing


class FrgVertexData():
    """ The functional renormalization group (fRG) on 1d spinless fermionic lattice model with flow of 
    density respond vertex. 
    The essntial data is fRG self-energy and density respond vertex enhenced local density.
    """

    def __init__(self):
        """ Create a new fRG data.

        _frg_vertexflow_data     the name of the all of the frg data during the flow
        _local_density           the name of the local density  
        _local_density_parallel  the name of the local density using the parallel compuation 
        """

        self._frg_vertexflow_data=0.0
        self._local_density=0.0
        self._local_density_parallel=0.0

    def get_frg_vertexflow_data(self):
        """Return the fRG self-energy in the end of the flow."""
        return self._frg_vertexflow_data

    def get_local_density(self):
        """Retrun the local density obtained from frg vertex method with specific position."""
        return self._local_density


    def vertex_solver(self,site,n,z,dt,u,v,du,mu,phi_v=0.0,phi_t=0.0,phi_u=0.0,t=1.0):
        """The solver generating local density using vertex flow method.

        Input Parameters:
        ----------
        site: integer
            specific position.
        n : integer
            The total system size.
        z : integer
            The number of sub-lattice.
        dt : float
            The magnitude of the hopping modulation.
        u : float
            The magnitude of the interaction.
        v : float
            The magnitude of onsite potential.
        du : float 
            The magnitude of modulation of the interaction.
        mu : float
            The chemical potential.
        phi_v, phi_t ,phi_u : float
            The phase of the onsite-potential, hopping and the interaction.
        t : float
            4t is the band-width.

        Output:
        ----------
        The fRG self-energy, vertex respond function and local density during the RG flow.
        """  
        t_i=t_m(n,z,phi_t,t,dt)
        v_i=v_m(n,z,phi_v,v)
        
        @njit()
        def diffeq_vertex(l, y):
            ## y are solutions of differential equations
            ## l is the flow parameter of differential equations
            ## single scale propagator for positive cutoff
        
            g_dia = np.zeros(n,np.complex128) 
            g_off = np.zeros(n-1,np.complex128)  
            tridiagonal=tri(l,y,v_i,t_i,mu,n)
            a=tridiagonal[0]
            b=tridiagonal[1]
                   
            inverse_tri=inv_tri(n,a,b)  
            g_off=inverse_tri[0]
            g_dia=inverse_tri[1] 
        

            diffeq = rg_eq_densityvertex(g_off,g_dia,n,z,phi_u,u,du,l,y)
                 
            return diffeq 

        #initial condition of the solution
        init=init_vertex(n=n,u=u,site=site)
    
        #starting and end points of the flow parameter
        flow=[1e8,1e-7]

        #relative accuracy in RK45 solver
        acc =1e-10

        sol = solve_ivp(diffeq_vertex,flow,init,method='RK45',rtol=acc,atol=1e-11)

        self._frg_vertexflow_data=sol
        self._local_density=sol.y[4*n,-1]

        return sol

   


