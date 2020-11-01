import numpy as np
from numpy import exp
from frg.greenfunction import FrgGreenData
from frg.free import FreeData



def search_mu_green(n,z, 
                    dt,u,du,v, 
                    pn,mu_upper,mu_lower,
                    phi_v=0.0,phi_t=0.0,phi_u=0.0,t=1.0):
    """ Using binary search algorithm to find the chemical potential away from half-filling.
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
    phi_v, phi_t ,phi_u : float
        The phase of the onsite-potential, hopping and the interaction.
    t : float
        4t is the band-width.

    Output:
        ----------
        The final chemical potential.
    
    """
    d=5
    target=pn
    mu_max=mu_upper
    mu_min=mu_lower
    mu_range=np.zeros(2,dtype=np.float64)
    mu_range[0]=mu_min
    mu_range[1]=mu_max

    p=FrgGreenData()
    p.frg_obc(n=n,z=z,dt=dt,u=u,v=v,du=du,mu=mu_max,
              phi_v=phi_v,phi_t=phi_t,phi_u=phi_u)
    p.local_density(n=n,z=z,dt=dt,u=u,v=v,du=du,mu=mu_max,
                    phi_v=phi_v,phi_t=phi_t,phi_u=phi_u)
    rho=p.get_local_density()
    pn_max=np.around(np.sum(rho),d)

    if pn_max==target:
        return mu_max
    else:
        pass
    
    p.frg_obc(n=n,z=z,dt=dt,u=u,v=v,du=du,mu=mu_min,phi_v=phi_v,phi_t=phi_t,phi_u=phi_u)
    p.local_density(n=n,z=z,dt=dt,u=u,v=v,du=du,mu=mu_min,
                    phi_v=phi_v,phi_t=phi_t,phi_u=phi_u)
    rho=p.get_local_density()
    pn_min=np.around(np.sum(rho),d)

    if pn_min==target:
        return mu_min
    else:
        pass

    if pn_min>target and pn_max>target:
        raise ValueError('Two arguments are both larger than the final mu.')
    if pn_min<target and pn_max<target:
        raise ValueError('Two arguments are both smaller than the final mu.')
    count=2

    while True:
        count+=1
        mu=(mu_max+mu_min)/2.0
        p.frg_obc(n=n,z=z,dt=dt,u=u,v=v,du=du,mu=mu,phi_v=phi_v,phi_t=phi_t,phi_u=phi_u)
        p.local_density(n=n,z=z,dt=dt,u=u,v=v,du=du,mu=mu,phi_v=phi_v,phi_t=phi_t,phi_u=phi_u)
        rho=p.get_local_density()
        if target==np.around(np.sum(rho),d):
            print(np.sum(rho))
            return mu
            break 
        if target>np.around(np.sum(rho),d):
            mu_min=mu
            pn_min=np.around(np.sum(rho),d)

        elif target<np.around(np.sum(rho),d):
            mu_max=mu
            pn_max=np.around(np.sum(rho),d)
        else:
            pass 


def dQdgamma(gamma,bc):
    
    """ 
    Input Parameters:
    ----------
    gamma : list of floats
        The list of the targeted phases of the modulation.
    bc :  list of floats
        The list of the targeted boundary charge with corresponding gamma.   

    Output:
        ----------
        Derivative of boundary charge with respect to the gamma.
    
    """
    b=np.zeros(int(len(bc))-1,dtype="float") 
    a=np.zeros(int(len(bc))-1,dtype="float") 
    for i in range(int(len(bc))-1):
        b[i]=(bc[i+1]-bc[i])/(gamma[i+1]-gamma[i])
        a[i]=gamma[i]
    return [a,b]

def generate_boundarycharge(n,z, 
                            dt,u,du,v, 
                            filling,mu,
                            phi_v=0.0,phi_t=0.0,phi_u=0.0,t=1.0):
    """ generating the boundary charge with FRG self-energy + green's function approach.
    
    
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
    filling: float
        The filling: total particle number/total system size.
    mu : float
        chemical potential.
    phi_v, phi_t ,phi_u : float
        The phase of the onsite-potential, hopping and the interaction.
    t : float
        4t is the band-width.

    Output:
        ----------
        The boundary charge.
    
    """
    p=FrgGreenData()
    d=5
    p.frg_obc(n=n,z=z,dt=dt,u=u,v=v,du=du,mu=mu,
              phi_v=phi_v,phi_t=phi_t,phi_u=phi_u)
    p.local_density(n=n,z=z,dt=dt,u=u,v=v,du=du,mu=mu,
                    phi_v=phi_v,phi_t=phi_t,phi_u=phi_u)
    rho=p.get_local_density()
    print('particle number:',np.around(np.sum(rho),d))
    #print(Ans)
    p.boundary_charge(n=n,filling=filling)
    p_bc=p.get_boundary_charge()
    #print('boundary charge:',p4)
    return p_bc
    #print('----------')


def eff_gap_z4(n,z,dt,u,v,du,mu,
            phi_v=0.0,phi_t=0.0,phi_u=0.0,ave=10):
    
    """ generating the effecitve FRG gap with FRG self-energy + green's function approach.
    
    
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
    filling: float
        The filling: total particle number/total system size.
    mu : float
        chemical potential.
    phi_v, phi_t ,phi_u : float
        The phase of the onsite-potential, hopping and the interaction.
    ave : int
        the numbers of the simplies for the bulk averaging.

    Output: list of floats
        ----------
        The FRG effecitve gap.
    
    """
    p=FrgGreenData()
    p.frg_obc(n=n,z=z,dt=dt,u=u,v=v,du=du,mu=mu,phi_v=phi_v,phi_t=phi_t,phi_u=phi_u)
    p.bulk_parameters(n=n,z=z,dt=dt,u=u,v=v,du=du,mu=mu,phi_v=phi_v,phi_t=phi_t,phi_u=phi_u,t=1.0,ave=ave)
    ans_bulk_parameters=p.get_bulk_parameters()
    ren_v=ans_bulk_parameters[0]
    ren_t=ans_bulk_parameters[1]
    p.bulk_gap_aah_z4(n=n,z=z,t_j=ren_t,v_j=ren_v)
    ans_gap=p.get_gap()
    return ans_gap

def envelope_f(n,k1,k2):
    """envelope function for boundary charge.
    
    Input:
    ------------------
    n : integer
        The total system size.  
    k1 : float
        inverse of the decay length of the enveloping function.
    k2 : float
        the position started the decay.
    Output:
    """
    f = np.zeros(int(n/2),dtype="float")
    for ii in range(int(n/2)):
        f[ii]=(1.0/(1.0+exp(-k1*(-(ii-(n/2.0-k2))))) )
    return f




def eff_boundary_charge_z4(n,z,dt,u,v,du,mu,filling,
                            phi_v=0.0,phi_t=0.0,phi_u=0.0,ave=10):
    """ generating the non-interacting boundary charge with FRG effecitve single particle parameters.
    
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
    filling: float
        The filling: total particle number/total system size.
    mu : float
        chemical potential.
    phi_v, phi_t ,phi_u : float
        The phase of the onsite-potential, hopping and the interaction.
    t : float
        4t is the band-width.

    Output:
        ----------
        The boundary charge.
    
    """
    p=FrgGreenData()
    p.frg_obc(n=n,z=z,dt=dt,u=u,v=v,du=du,mu=mu,phi_v=phi_v,phi_t=phi_t,phi_u=phi_u)
    p.bulk_parameters(n=n,z=z,dt=dt,u=u,v=v,du=du,mu=mu,phi_v=phi_v,phi_t=phi_t,phi_u=phi_u,t=1.0,ave=ave)
    ans_bulk_parameters=p.get_bulk_parameters()
    ren_v=ans_bulk_parameters[0]
    ren_t=ans_bulk_parameters[1]
    print('eff v:',ren_v)
    print('eff t:',ren_t)
    pp=FreeData()
    pp.effective_local_density(n=n,z=z,ren_t=ren_t,ren_v=ren_v,mu=mu) 
    #print('local density fine')
    ans=pp.boundary_charge(n=n,filling=filling)   

    return ans