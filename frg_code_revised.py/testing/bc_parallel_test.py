import numpy as np
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import sys 

sys.path.append('../..')

from frg.greenfunction import FrgGreenData

from frg.parallel import find_mu_aah_z4_parallel,find_mu_mod_u_z4_parallel

from frg.tool import loglogder


sys.path.append('project1')

from parameter.parameters_bc_3qrt_filling import *
p1=125
p2=150
p3=175
p4=200
p5=250

a=find_mu_mod_u_z4_parallel(n_phi=NNN,n=N,z=Z,dt=DT,u=U,v=V,du=DU*p1,pn=TARGET,mu_upper=3.0,mu_lower=-1e-6)
b=find_mu_mod_u_z4_parallel(n_phi=NNN,n=N,z=Z,dt=DT,u=U,v=V,du=DU*p2,pn=TARGET,mu_upper=3.0,mu_lower=-1e-6)
c=find_mu_mod_u_z4_parallel(n_phi=NNN,n=N,z=Z,dt=DT,u=U,v=V,du=DU*p3,pn=TARGET,mu_upper=4.0,mu_lower=-1e-6)
d=find_mu_mod_u_z4_parallel(n_phi=NNN,n=N,z=Z,dt=DT,u=U,v=V,du=DU*p4,pn=TARGET,mu_upper=4.0,mu_lower=-1e-6)
e=find_mu_mod_u_z4_parallel(n_phi=NNN,n=N,z=Z,dt=DT,u=U,v=V,du=DU*p5,pn=TARGET,mu_upper=5.0,mu_lower=-1e-6)

mu_list_a=a
mu_list_b=b
mu_list_c=c
mu_list_d=d
mu_list_e=e



f=0.25
bc_list_a=np.zeros(NNN,dtype=np.float64)
bc_list_b=np.zeros(NNN,dtype=np.float64)
bc_list_c=np.zeros(NNN,dtype=np.float64)
bc_list_d=np.zeros(NNN,dtype=np.float64)
bc_list_e=np.zeros(NNN,dtype=np.float64)

for i in range(NNN):
    phi=2.0*np.pi*i/NNN+0.001

    bc_a=generate_boundarycharge(n=N,z=Z,dt=DT,u=U,du=DU*p,v=V,filling=f,
                                  mu=mu_list_a[i],phi_u=phi)
    bc_b=generate_boundarycharge(n=N,z=Z,dt=DT,u=U,du=DU*pp,v=V,filling=f,
                                  mu=mu_list_b[i],phi_u=phi)
    bc_c=generate_boundarycharge(n=N,z=Z,dt=DT,u=U,du=DU*ppp,v=V,filling=f,
                                  mu=mu_list_c[i],phi_u=phi)
    bc_d=generate_boundarycharge(n=N,z=Z,dt=DT,u=U,du=DU*pppp,v=V,filling=f,
                                  mu=mu_list_d[i],phi_u=phi)
    bc_e=generate_boundarycharge(n=N,z=Z,dt=DT,u=U,du=DU*ppppp,v=V,filling=f,
                                  mu=mu_list_e[i],phi_u=phi)
    
 
    bc_list_a[i]=bc_a
    bc_list_b[i]=bc_b
    bc_list_c[i]=bc_c
    bc_list_d[i]=bc_d
    bc_list_e[i]=bc_e
    print('==========')



np.save('frg_data/mu_3qrt_dU=%g.npy'%(DU*p),mu_list_a)
np.save('frg_data/bc_3qrt_dU=%g.npy'%(DU*p),bc_list_a)

np.save('frg_data/mu_3qrt_dU=%g.npy'%(DU*pp),mu_list_b)
np.save('frg_data/bc_3qrt_dU=%g.npy'%(DU*pp),bc_list_b)

np.save('frg_data/mu_3qrt_dU=%g.npy'%(DU*ppp),mu_list_c)
np.save('frg_data/bc_3qrt_dU=%g.npy'%(DU*ppp),bc_list_c)

np.save('frg_data/mu_3qrt_dU=%g.npy'%(DU*pppp),mu_list_d)
np.save('frg_data/bc_3qrt_dU=%g.npy'%(DU*pppp),bc_list_d)

np.save('frg_data/mu_3qrt_dU=%g.npy'%(DU*ppppp),mu_list_e)
np.save('frg_data/bc_3qrt_dU=%g.npy'%(DU*ppppp),bc_list_e)
