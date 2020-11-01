import numpy as np
import sys 
sys.path.append('../..')
from tool.frg_aah_tool import eff_gap_z4,search_mu_green
from tool.frg_aah_greenfunction import FrgGreenData
sys.path.append('project2')
from parameter.three_qrt_filling import N_GAP,N,Z,DT,V,DU_0,PHI_V,PHI_T,PHI_U,U_0,N_U,TARGET

frg_mu=np.zeros([N_U,N_GAP],dtype=np.float64)
free_mu=np.zeros([N_U,N_GAP],dtype=np.float64)

frg_bulk_gap=np.zeros([N_U,N_GAP,3],dtype=np.float64)
free_bulk_gap=np.zeros([N_U,N_GAP,3],dtype=np.float64)

print('U:',U_0)
print('dU:',DU_0)
print('v:',V)
print('dt:',DT)

mu_free=search_mu_green(n=N,z=Z, 
                        dt=DT,u=0.0,du=DU_0,v=V, 
                        pn=TARGET,mu_upper=2.0,mu_lower=0.5,
                        phi_v=PHI_V,phi_t=PHI_T,phi_u=PHI_U,t=1.0)

for ii in range(N_GAP):
        
    delta=0.005*ii
    DU=DU_0+delta
    sol_gap_free=eff_gap_z4(n=N,z=Z,dt=DT,u=0.0,v=V,du=DU,mu=mu_free,
                            phi_v=PHI_V,phi_t=PHI_T,phi_u=PHI_U,ave=10)        
    for i in range(N_U):
        free_bulk_gap[i,ii,:]=sol_gap_free
        free_mu[i,ii]=mu_free

for i in range(N_U):
    U=U_0*i
    
    for ii in range(N_GAP):
        delta=0.005*ii
        DU=DU_0+delta
        
        print('U:',U)
        print('dU:',DU)
        print('v:',V)
        print('dt:',DT)
        mu_frg=search_mu_green(n=N,z=Z, 
                           dt=DT,u=U,du=DU,v=V, 
                           pn=TARGET,mu_upper=2.0,mu_lower=1.2,
                           phi_v=PHI_V,phi_t=PHI_T,phi_u=PHI_U,t=1.0)
        p=FrgGreenData()
        p.frg_obc(n=N,z=Z,dt=DT,u=U,v=V,du=DU,mu=mu_frg,phi_v=PHI_V,phi_t=PHI_T,phi_u=PHI_U)
        ans1=p.local_density(n=N,z=Z,dt=DT,u=U,v=V,du=DU,mu=mu_frg,phi_v=PHI_V,phi_t=PHI_T,phi_u=PHI_U)
        print('---------')
        print(ans1.y[:,-1],np.sum(ans1.y[:,-1]))

        sol_gap_frg=eff_gap_z4(n=N,z=Z,
                               dt=DT,u=U,v=V,du=DU,mu=mu_frg,
                               phi_v=PHI_V,phi_t=PHI_T,phi_u=PHI_U,ave=10)

        frg_bulk_gap[i,ii,:]=sol_gap_frg
        frg_mu[i,ii]=mu_frg
        print('frg gap:',sol_gap_frg)

    print('frg mu:',mu_frg)
    print(i)
    print('===========')

np.save('data/frg_gap_3qrt_dU=%g_U=%g.npy'%(DU_0,U_0),frg_bulk_gap)
np.save('data/free_gap_3qrt_dU=%g_U=%g.npy'%(DU_0,U_0),free_bulk_gap)

np.save('data/mu_gap_3qrt_dU=%g_U=%g.npy'%(DU_0,U_0),frg_mu)
np.save('data/mu_free_gap_3qrt_dU=%g_U=%g.npy'%(DU_0,U_0),free_mu)
