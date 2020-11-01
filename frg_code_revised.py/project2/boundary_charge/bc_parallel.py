import numpy as np
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import sys 

sys.path.append('../..')

from frg.tool import generate_boundarycharge
from frg.parallel import find_mu_aah_z4_parallel,find_mu_mod_u_z4_parallel
sys.path.append('project2/boundary_charge')

from parameter.parameters_bc_3qrt_filling import *



mu_list_a=find_mu_mod_u_z4_parallel(n_phi=NNN,n=N,z=Z,dt=DT,u=U,v=V,du=DU*P1,pn=TARGET,mu_upper=MU_UPPER_1,mu_lower=MU_LOWER_1)
mu_list_b=find_mu_mod_u_z4_parallel(n_phi=NNN,n=N,z=Z,dt=DT,u=U,v=V,du=DU*P2,pn=TARGET,mu_upper=MU_UPPER_2,mu_lower=MU_LOWER_2)
mu_list_c=find_mu_mod_u_z4_parallel(n_phi=NNN,n=N,z=Z,dt=DT,u=U,v=V,du=DU*P3,pn=TARGET,mu_upper=MU_UPPER_3,mu_lower=MU_LOWER_3)
mu_list_d=find_mu_mod_u_z4_parallel(n_phi=NNN,n=N,z=Z,dt=DT,u=U,v=V,du=DU*P4,pn=TARGET,mu_upper=MU_UPPER_4,mu_lower=MU_LOWER_4)
mu_list_e=find_mu_mod_u_z4_parallel(n_phi=NNN,n=N,z=Z,dt=DT,u=U,v=V,du=DU*P5,pn=TARGET,mu_upper=MU_UPPER_5,mu_lower=MU_LOWER_5)


bc_list_a=np.zeros(NNN,dtype=np.float64)
bc_list_b=np.zeros(NNN,dtype=np.float64)
bc_list_c=np.zeros(NNN,dtype=np.float64)
bc_list_d=np.zeros(NNN,dtype=np.float64)
bc_list_e=np.zeros(NNN,dtype=np.float64)

for i in range(NNN):
    phi=2.0*np.pi*i/NNN+0.001

    bc_a=generate_boundarycharge(n=N,z=Z,dt=DT,u=U,du=DU*P1,v=V,filling=FILLING,
                                  mu=mu_list_a[i],phi_u=phi)
    bc_b=generate_boundarycharge(n=N,z=Z,dt=DT,u=U,du=DU*P2,v=V,filling=FILLING,
                                  mu=mu_list_b[i],phi_u=phi)
    bc_c=generate_boundarycharge(n=N,z=Z,dt=DT,u=U,du=DU*P3,v=V,filling=FILLING,
                                  mu=mu_list_c[i],phi_u=phi)
    bc_d=generate_boundarycharge(n=N,z=Z,dt=DT,u=U,du=DU*P4,v=V,filling=FILLING,
                                  mu=mu_list_d[i],phi_u=phi)
    bc_e=generate_boundarycharge(n=N,z=Z,dt=DT,u=U,du=DU*P5,v=V,filling=FILLING,
                                  mu=mu_list_e[i],phi_u=phi)
    
 
    bc_list_a[i]=bc_a
    bc_list_b[i]=bc_b
    bc_list_c[i]=bc_c
    bc_list_d[i]=bc_d
    bc_list_e[i]=bc_e
    print('==========')
    
sys.path.append('project2/boundary_charge')

np.save('data/mu_3qrt_dU=%g.npy'%(DU*P1),mu_list_a)
np.save('data/bc_3qrt_dU=%g.npy'%(DU*P1),bc_list_a)

np.save('data/mu_3qrt_dU=%g.npy'%(DU*P2),mu_list_b)
np.save('data/bc_3qrt_dU=%g.npy'%(DU*P2),bc_list_b)

np.save('data/mu_3qrt_dU=%g.npy'%(DU*P3),mu_list_c)
np.save('data/bc_3qrt_dU=%g.npy'%(DU*P3),bc_list_c)

np.save('data/mu_3qrt_dU=%g.npy'%(DU*P4),mu_list_d)
np.save('data/bc_3qrt_dU=%g.npy'%(DU*P4),bc_list_d)

np.save('data/mu_3qrt_dU=%g.npy'%(DU*P5),mu_list_e)
np.save('data/bc_3qrt_dU=%g.npy'%(DU*P5),bc_list_e)

