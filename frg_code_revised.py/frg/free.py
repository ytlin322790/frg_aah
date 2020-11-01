from numba import njit
from numba import njit,int32,float64
import numpy as np
from numpy import pi,exp
from numpy import linalg as LA
from scipy.integrate import solve_ivp
from tool.tool_aah_solver import t_m,v_m,inv_tri




def t_eff_z4(n,z,ren_t):
    """ generating the all matrix elements of the effecitve hopping matrix
    
    
    Input Parameters:
    ----------
    n : integer
        The total system size.
    z : integer
        The number of sub-lattice.
    ren_t : float
        The effective hopping in one unit cell. 

    Output: list of floats
        ----------
        The matrix elements of the hopping. 
    
    """ 
    t_m=np.zeros(n,np.float64)   
    for m in range(0,n,z): 
        for j in range(z):
            t_m[m+j]=ren_t[j]
    return t_m

def v_eff_z4(n,z,ren_v):
    """ generating the all matrix elements of the matrix of effecitve on-site potential.

    Input Parameters:
    ----------
    n : integer
        The total system size.
    z : integer
        The number of sub-lattice.
    ren_t : float
        The effective hopping in one unit cell. 

    Output: list of floats
        ----------
        The matrix elements of the hopping. 
    
    """ 
    v_m=np.zeros(n,np.float64) 

    for m in range(0,n,z): 
        for j in range(z):
            v_m[m+j]=ren_v[j]
    return v_m

class FreeData():
    """ Data for non-interacting AAH model 
    """

    def __init__(self):
        """ Create a new fRG data.

        _local_density     the name of the local density for non-interacting AAH model
        _boundary_charge   the name of the boundary charge for given local density 
        _spectrum          the name of the spectrum of the non-interacting AAH model
        """
        self._local_density=0.0
        self._boundary_charge=0.0
        self._spectrum=0.0


    def get_local_density(self):
        """Retrun the local density obtained from green function method."""
        return self._local_density

    def get_boundary_charge(self):
        """Retrun the boundary charge obtained green function method."""
        return self._boundary_charge

    def get_spectrum(self):
        """Retrun the boundary charge obtained green function method."""
        return self._spectrum   
     
    def local_density(self,n,z,dt,v,mu,
                      phi_v=0.0,phi_t=0.0,phi_u=0.0,t=1.0):
        """ The solver generating local density using green function method.

        Input Parameters:
        ----------
        n : integer
            The total system size.
        z : integer
            The number of sub-lattice.
        dt : float
            The magnitude of the hopping modulation.
        v : float
            The magnitude of onsite potential.
        mu : float
            The chemical potential.
        phi_v, phi_t ,phi_u : float
            The phase of the onsite-potential, hopping and the interaction.
        t : float
            4t is the band-width.

        Output:
        ----------
        The fRG self-energy during the RG flow.
        """    
        v_i=v_m(n,z,phi_v,v)
        t_i=t_m(n,z,phi_t,t,dt)


        @njit()
        def diffeq_density(l,n,t_i,v_i,mu):
            """Generator of right hand side of the diffential equation of local density.

            Input:
            ------------
            
            l : float
            the parameter of differential equations
            n : integer
            the number of total system size.
            t_i : set
            one period of the hopping parameters 
            v_i : set 
            one period of the onsite potential
            mu: float
            chemical potential


            Output:
            ------------
            right hand side of the RG equations
            """

            g_dia=np.zeros(n,dtype=np.complex128) 
            a=np.zeros(n,dtype=np.complex128)
            b=np.zeros(n-1,dtype=np.complex128)
            for i in range(n-1):
                a[i]=1.0j*l+mu-v_i[i]
                b[i]=t_i[i]
            a[n-1]=1.0j*l+mu-v_i[n-1]

            inverse_tri=inv_tri(n,a,b)  
            g_dia=inverse_tri[1] 
            diffeq=np.zeros(n,dtype=np.float64)
            for i in range(n):        
                diffeq[i]=(1.0/pi)*g_dia[i].real  
            return diffeq 

        def diff_eq_density(l,y):  
            diffeq=diffeq_density(l,n,t_i,v_i,mu)               
            return diffeq   

        #initial condition
        init=np.zeros(n,dtype="float")
        for i in range(n):
            init[i]=0.5     

        #flow parameter
        flow=[1e-20,1e20] 

        # relative accuarcy of RK45 solver
        acc =2.220446049250313e-14

        density = solve_ivp(diff_eq_density,flow,init,method='RK45',rtol=acc,atol=1e-15)  

        self._local_density=density.y[:,-1]
        
        return density

    def boundary_charge(self,n,filling,k1=0.15,k2=100):
        """ The solver generating boundary charge for given local density.

        Input Parameters:
        ----------
        n : integer
            The total system size.
        filling : float
            The filling of total system: total particle/total system size.
        k1 : float
            Inverse of the decay length of envelope function.
        k2 : float
            Starting position of the decay.


        Output:
        ----------
        boundary charge: float
        """    
        def envelope_f(n,k1,k2):
            "envelope function"
            f = np.zeros(int(n/2),dtype="float") 
            for ii in range(int(n/2)):
                f[ii]=(1.0/(1.0+exp(-k1*(-(ii-(n/2.0-k2))))) )
            return f

        f = envelope_f(n,k1=k1,k2=k2)
        local_density=self.get_local_density()
        bc=np.sum((local_density[0:int(n/2):1]-filling)*f) 
        #print(bc)
        self._boundary_charge=bc

        return bc

    def spectral_aah_free(self,n,z,dt,v,mu,phi_t,phi_v,pn,t=1.0):
        """ The solver generating spectrum of non-interacting AAH model using exact diagonization.

        Input Parameters:
        ----------
        n : integer
            The total system size.
        z : integer
            The number of sub-lattice.
        dt : float
            The magnitude of the hopping modulation.
        v : float
            The magnitude of onsite potential.
        mu : float
            The chemical potential.
        phi_v, phi_t : float
            The phase of the onsite-potential, and hopping.
        pn : float 
            The total particle number.
        t : float
            4t is the band-width.

        Output:
        ----------
        The fRG self-energy during the RG flow.
        """   

        H=np.zeros((n,n),dtype=np.float64) # effective AAH Hamiltonian

        v_i=v_m(n,z,phi_v,v)
        t_i=t_m(n,z,phi_t,t,dt)
        for i in range(n-1):
            H[i,i]=-mu+v_i[i] 
            H[i,i+1]=H[i+1,i]=-t_i[i]       
        H[n-1,n-1]=-mu+v_i[n-1] 
 
        w,v=LA.eigh(H)
        #v[:,i] eignevector with i eigen-value

        density=np.zeros(n,dtype=np.float64)
        for i in range(n):
            for ii in range(n):
                density[i]=density[i]+np.vdot(v[i, ii],v[i,ii])
        ans=pn/n        
        print('this is %g filling spectrum and density for non-interacting AAH Z=%g model'%(ans,z))
        self._spectrum=[w,v]
        return [w,v,density] 

    def effective_aah_local_density_ed(self,n,z,ren_t,ren_v,mu,t=1.0):
        """ The solver generating spectrum of non-interacting AAH model using exact diagonization.

        Input Parameters:
        ----------
        n : integer
            The total system size.
        z : integer
            The number of sub-lattice.
        ren_t : set of floats
            Effective hopping terms in unit cell.
        ren_v : set of floats
            Effective onsite potential in unit cell.
        mu : float
            The chemical potential.

        Output:
        ----------
        The fRG self-energy during the RG flow.
        """   

        H=np.zeros((n,n),dtype=np.float64) # effective AAH Hamiltonian
        for i in range(n-1):
            H[i,i]=-mu+ren_v[i] 
            H[i,i+1]=H[i+1,i]=-ren_t[i]       
        H[n-1,n-1]=-mu+ren_v[n-1] 
 
        w,v=LA.eigh(H)
        #v[:,i] eignevector with i eigen-value

        density=np.zeros(n,dtype=np.float64)
        for i in range(n):
            for ii in range(n):
                density[i]=density[i]+np.vdot(v[i, ii],v[i,ii])
        
        self._spectrum=[w,v]
        return [w,v,density]

    def effective_local_density(self,n,z,ren_t,ren_v,mu):
        """ The solver generating local density using green function method.

        Input Parameters:
        ----------
        n : integer
            The total system size.
        z : integer
            The number of sub-lattice.
        ren_t : set of floats
            Effective hopping terms in unit cell.
        ren_v : set of floats
            Effective onsite potential in unit cell.
        mu : float
            The chemical potential.

        Output:
        ----------
        Local density in effective single particle picture.
        """    

        ren_t_eff=t_eff_z4(n,z,ren_t)
        ren_v_eff=v_eff_z4(n,z,ren_v)
        @njit()
        def diffeq_density(l,n,ren_t_eff,ren_v_eff,mu):
            """Generator of right hand side of the diffential equation of local density.

            Input:
            ------------
            
            l : float
                the parameter of differential equations
            n : integer
                the number of total system size.
            ren_t : set of floats
                Effective hopping terms in unit cell.
            ren_v : set of floats
                Effective onsite potential in unit cell.
            mu: float
                chemical potential.


            Output:
            ------------
            right hand side of the RG equations
            """
            g_dia=np.zeros(n,dtype=np.complex128) 
            a=np.zeros(n,dtype=np.complex128)
            b=np.zeros(n-1,dtype=np.complex128)
            for i in range(n-1):
                
                a[i]=1.0j*l+mu-ren_v_eff[i]
                b[i]=ren_t_eff[i]
            a[n-1]=1.0j*l+mu-ren_v_eff[n-1]

            inverse_tri=inv_tri(n,a,b)  
            g_dia=inverse_tri[1] 
            diffeq=np.zeros(n,dtype=np.float64)
            for i in range(n):        
                diffeq[i]=(1.0/pi)*g_dia[i].real  
            return diffeq 


        def diff_eq_density(l,y):  
            diffeq=diffeq_density(l,n,ren_t_eff,ren_v_eff,mu)               
            return diffeq   

        #initial condition
        init=np.zeros(n,dtype="float")
        for i in range(n):
            init[i]=0.5     
        #flow parameter
        flow=[1e-10,1e10] 
        # relative accuarcy of RK45 solver
        acc =2.220446049250313e-14

        density = solve_ivp(diff_eq_density,flow,init,method='RK45',rtol=acc,atol=1e-15)  

        self._local_density=density.y[:,-1]
        
        return density

 

    def effective_boundary_charge(self,n,filling,k1=0.15,k2=100):
        """ The solver generating boundary charge for given local density.

        Input Parameters:
        ----------
        n : integer
            The total system size.
        filling : float
            The filling of total system: total particle/total system size.
        k1 : float
            Inverse of the decay length of envelope function.
        k2 : float
            Starting position of the decay.


        Output:
        ----------
        boundary charge: float
        """    
        def envelope_f(n,k1,k2):
            "envelope function"
            f=np.zeros(int(n/2),dtype="float") 
            for ii in range(int(n/2)):
                f[ii]=(1.0/(1.0+exp(-k1*(-(ii-(n/2.0-k2))))) )
            return f
        f=envelope_f(n,k1=k1,k2=k2)
        local_density=self.get_local_density()
        bc=np.sum((local_density[0:int(n/2):1]-filling)*f) 
        self._boundary_charge=bc

        return bc