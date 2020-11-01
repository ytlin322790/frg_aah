import numpy as np
import sys 
sys.path.append('../..')
from frg.tool import eff_gap_z4,search_mu_green
sys.path.append('project1')
from parameter.parameters_gap_3qrt_filling_trial import *


frg_mu=np.zeros([N_U,N_GAP],dtype=np.float64)
free_mu=np.zeros([N_U,N_GAP],dtype=np.float64)

frg_bulk_gap=np.zeros([N_U,N_GAP,3],dtype=np.float64)
free_bulk_gap=np.zeros([N_U,N_GAP,3],dtype=np.float64)


print(V_0,DT_0)
for i in range(N_U):
    U=U_0*i

    mu_frg=search_mu_green(n=N,z=Z, 
                           dt=DT_0,u=U,du=DU,v=V_0, 
                           pn=TARGET,mu_upper=2.0,mu_lower=0.5,
                           phi_v=PHI_V,phi_t=PHI_T,phi_u=PHI_U,t=1.0)

    mu_free=search_mu_green(n=N,z=Z, 
                            dt=DT_0,u=0.0,du=DU,v=V_0, 
                            pn=TARGET,mu_upper=2.0,mu_lower=0.5,
                            phi_v=PHI_V,phi_t=PHI_T,phi_u=PHI_U,t=1.0)
    

    for ii in range(N_GAP):

        delta=0.00000001*ii
        V=V_0+delta
        DT=DT_0+delta

        sol_gap_free=eff_gap_z4(n=N,z=Z,dt=DT,u=0.0,v=V,du=DU,mu=mu_free,
                                phi_v=PHI_V,phi_t=PHI_T,phi_u=PHI_U,ave=10)

        sol_gap_frg=eff_gap_z4(n=N,z=Z,
                            dt=DT,u=U,v=V,du=DU,mu=mu_frg,
                            phi_v=PHI_V,phi_t=PHI_T,phi_u=PHI_U,ave=10)
        

   
        free_bulk_gap[i,ii,:]=sol_gap_free
        frg_bulk_gap[i,ii,:]=sol_gap_frg
        frg_mu[i,ii]=mu_frg
        free_mu[i,ii]=mu_free

    print('mu_free',mu_free)
    print('mu_frg_',mu_frg)
    print('--------')
    print(i)




np.save('data/frg_gap_3qrt_v=%g_dt=%g.npy'%(V_0,DT_0),frg_bulk_gap)
np.save('data/free_gap_3qrt_v=%g_dt=%g.npy'%(V_0,DT_0),free_bulk_gap)

np.save('data/mu_gap_3qrt_v=%g_dt=%g.npy'%(V_0,DT_0),frg_mu)
np.save('data/mu_free_gap_3qrt_v=%g_dt=%g.npy'%(V_0,DT_0),free_mu)

