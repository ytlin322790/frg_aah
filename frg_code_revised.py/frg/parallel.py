import numpy as np
from tool.frg_aah_tool import search_mu_green
from tool.frg_aah_kspace import frg_vertex_z4_bulk_solver
from functools import partial
import multiprocessing


def find_mu_aah_z4(n,z,dt,u,du,v,pn,mu_upper,mu_lower,phi):
    """ Using binary search algorithm to find the chemical potential away from half-filling
        for the data of boudnary charge as a function of gamma for the AAH model.
    The data for the local denisty is calculated from fRG self-energy plus single particle green's function.
    
    
    Input Parameters:
    ----------
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
    mu_upper : float
        Upper limit for trial chemical potential.
    mu_lower : float
        lower limit for trial chemical potential.
    phi : float
        The phase of the onsite-potential, hopping and the interaction.
        phi=phi_t=phi_v
        
    Output:
        ----------
        The final chemical potential.
    
    """
    mu=search_mu_green(n=n,z=z, 
                       dt=dt,u=u,du=du,v=v, 
                       pn=pn,mu_upper=mu_upper,mu_lower=mu_lower,
                       phi_v=phi,phi_t=phi)

    return mu   


def find_mu_aah_z4_parallel(n_phi,n,z,dt,u,v,du,pn,mu_upper,mu_lower):
    """ Parallel computing the boudnary charge as a function of gamma for the AAH model with Z=4.
    The data for the local denisty is calculated from fRG self-energy plus single particle green's function.
    
    Input Parameters:
    ----------
    n_phi: integer
        number of the gamma.
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
    mu_upper : float
        Upper limit for trial chemical potential.
    mu_lower : float
        lower limit for trial chemical potential.

    Output:
        ----------
        The list final chemical potential.
    
    """
    phi_list=np.zeros(n_phi,dtype=np.float64)
    for i in range(n_phi):
        phi=2.0*np.pi*i/n_phi+0.001
        phi_list[i]=phi

    mu_list=np.zeros(n_phi,dtype=np.float64)
    
    cores=multiprocessing.cpu_count()
    pool=multiprocessing.Pool(processes=cores)
    func=partial(find_mu_aah_z4,n,z,dt,u,du,v,pn,mu_upper,mu_lower)
    result=pool.map(func,phi_list)
    print(result)
    for i in range(n_phi):
        mu_list[i]=result[i]
        print(result[i])
    return mu_list



def find_mu_aah_z4_gap(n,z,dt,u,du,v,pn,mu_upper,mu_lower,phi_v,phi_t,delta):
    """ Using binary search algorithm to find the chemical potential away from half-filling
        for the data of boudnary charge as a function of gamma for the AAH model.
    The data for the local denisty is calculated from fRG self-energy plus single particle green's function.
    
    
    Input Parameters:
    ----------
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
    mu_upper : float
        Upper limit for trial chemical potential.
    mu_lower : float
        lower limit for trial chemical potential.
    phi_v,phi_t : float
        The phase of the onsite-potential, hopping respectively.
        
        
    Output:
        ----------
        The final chemical potential.
    
    """
    v_0=v+delta
    dt_0=dt+delta
    mu=search_mu_green(n=n,z=z, 
                       dt=dt_0,u=u,du=du,v=v_0,
                       pn=pn,mu_upper=mu_upper,mu_lower=mu_lower,
                       phi_v=phi_v,phi_t=phi_t)

    return mu   


def find_mu_aah_z4_gap_parallel(n_scan,n,z,dt,u,v,du,pn,mu_upper,mu_lower,phi_v,phi_t):
    """ Parallel computing the boudnary charge as a function of gamma for the AAH model with Z=4.
    The data for the local denisty is calculated from fRG self-energy plus single particle green's function.
    
    Input Parameters:
    ----------
    n_phi: integer
        number of the gamma.
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
    mu_upper : float
        Upper limit for trial chemical potential.
    mu_lower : float
        lower limit for trial chemical potential.
    phi_v,phi_t: float
        phase of the modulation.

    Output:
        ----------
        The list final chemical potential.
    
    """
    delta_list=np.zeros(n_scan,dtype=np.float64)
    for i in range(n_scan):
        delta=0.000001*i
        delta_list[i]=delta

    mu_list=np.zeros(n_scan,dtype=np.float64)
    
    cores=multiprocessing.cpu_count()
    pool=multiprocessing.Pool(processes=cores)

    
    func=partial(find_mu_aah_z4,n,z,dt,u,du,v,pn,delta,mu_upper,mu_lower,phi_v,phi_t)
    result=pool.map(func,delta_list)
    print(result)
    for i in range(n_scan):
        mu_list[i]=result[i]
        print(result[i])
    return mu_list



def find_mu_mod_u_z4(n,z,dt,u,du,v,pn,mu_upper,mu_lower,phi):
    """ Using binary search algorithm to find the chemical potential away from half-filling
        for the data of boudnary charge as a function of gamma for the modulated interactiong model with Z=4.
    The data for the local denisty is calculated from fRG self-energy plus single particle green's function.
    
    Input Parameters:
    ----------
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
    mu_upper : float
        Upper limit for trial chemical potential.
    mu_lower : float
        lower limit for trial chemical potential.
    phi : float
        The phase of the onsite-potential, hopping and the interaction.
        phi=phi_u


    Output:
        ----------
        The final chemical potential.
    
    """
    mu=search_mu_green(n=n,z=z, 
               dt=dt,u=u,du=du,v=v, 
               pn=pn,mu_upper=mu_upper,mu_lower=mu_lower,
               phi_u=phi)
    return mu   


def find_mu_mod_u_z4_parallel(n_phi,n,z,dt,u,v,du,pn,mu_upper,mu_lower):
    """ Parallel computing the boudnary charge as a function of gamma for the modulated interactiong model with Z=4.
    The data for the local denisty is calculated from fRG self-energy plus single particle green's function.
    
    Input Parameters:
    ----------
    n_phi: integer
        number of the gamma.
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
    mu_upper : float
        Upper limit for trial chemical potential.
    mu_lower : float
        lower limit for trial chemical potential.

    Output:
        ----------
        The list final chemical potential.
    
    """
    phi_list=np.zeros(n_phi,dtype=np.float64)
        phi=2.0*np.pi*i/n_phi+0.001
        phi_list[i]=phi

    mu_list=np.zeros(n_phi,dtype=np.float64)
    
    cores=multiprocessing.cpu_count()
    pool=multiprocessing.Pool(processes=cores)
    func=partial(find_mu_mod_u_z4,n,z,dt,u,du,v,pn,mu_upper,mu_lower)
    result=pool.map(func,phi_list)
    print(result)
    for i in range(n_phi):
        mu_list[i]=result[i]
        print(result[i])
    return mu_list




def fun_aah_z4(z,dt,u,du,v,mu,phi_v,phi_t,phi_u,site):
    """ The function generating the local density from FRG with density respond vertex.
    
    
    Input Parameters:
    ----------
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
    phi_v,phi_t,phi_u : float
        The phase of the onsite-potential, hopping and the interaction.
    site : integer
        Targeted sublattic index.
        
    Output:
        ----------
        The local density for given site.
    
    """
    ans=frg_vertex_z4_bulk_solver(site,z,dt,v,mu,u,du,phi_v,phi_t,phi_u)
    return ans.y[-1,-1]


def kspace_vertex_aah_z4_parallel(z,dt,u,v,du,phi_v,phi_t,phi_u,mu):
    """ Parallel computing local density in PBC.
    
    Input Parameters:
    ----------
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
    phi_v, phi_t, phi_u : float
        The phase of the onsite-potential, hopping and the interaction.
    mu : float
        The chemical potential.

    Output:
        ----------
        The list local density.
    
    """
    density_list=np.zeros(z,dtype=np.float64)


    unit_cell_list=np.zeros(z,dtype=np.int32)
    unit_cell_list[0]=0
    unit_cell_list[1]=1
    unit_cell_list[2]=2
    unit_cell_list[3]=3
    
    
    cores=multiprocessing.cpu_count()
    pool=multiprocessing.Pool(processes=cores)
    func=partial(fun_aah_z4,z,dt,u,du,v,mu,phi_v,phi_t,phi_u)
    result=pool.map(func,unit_cell_list)
    for i in range(z):
        density_list[i]=result[i]
        print(density_list[i])
    return density_list