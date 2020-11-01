import numpy as np
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import sys 
sys.path.append('..')
from frg.greenfunction import FrgGreenData
from frg.tool import eff_gap_z4,search_mu_green,generate_boundarycharge,dQdgamma
from frg.direivative import log_log_der

d=4
mu_list=np.zeros([NN,NNN],dtype=np.float64)
qb_list=np.zeros([NN,NNN],dtype=np.float64)
phi_list=np.zeros(NNN,dtype=np.float64)
mu_list=np.zeros([NN,NNN],dtype=np.float64)
qb_list=np.zeros([NN,NNN],dtype=np.float64)
phi_list=np.zeros(NNN,dtype=np.float64)
d=4
f=0.75

for i in range(NNN):
    
    print(i)
    
    phi=2.0*np.pi*i/NNN+0.001

    phi_list[i]=phi/(2.0*np.pi)
    
    pp=5.0
    ppp=10.0
    pppp=50.0
    ppppp=250.0
    p_mu=search_mu_green(n=N,z=Z, 
                        dt=DT,u=U,du=DU,v=V, 
                        pn=3000.0,mu_upper=5,mu_lower=1e-6,
                        phi_v=phi,phi_t=phi)
    pp_mu=search_mu_green(n=N,z=Z, 
                        dt=DT*pp,u=U,du=DU,v=V*pp, 
                        pn=3000.0,mu_upper=5,mu_lower=1e-6,
                        phi_v=phi,phi_t=phi)
    ppp_mu=search_mu_green(n=N,z=Z, 
                        dt=DT*ppp,u=U,du=DU,v=V*ppp, 
                        pn=3000.0,mu_upper=5,mu_lower=1e-6,
                        phi_v=phi,phi_t=phi)
    pppp_mu=search_mu_green(n=N,z=Z, 
                        dt=DT*pppp,u=U,du=DU,v=V*pppp, 
                        pn=3000.0,mu_upper=5,mu_lower=1e-6,
                        phi_v=phi,phi_t=phi)
    ppppp_mu=search_mu_green(n=N,z=Z, 
                        dt=DT*ppppp,u=U,du=DU,v=V*ppppp, 
                        pn=3000.0,mu_upper=5,mu_lower=1e-6,
                        phi_v=phi,phi_t=phi)                        

    mu_list[0,i]=p_mu
    mu_list[1,i]=pp_mu
    mu_list[2,i]=ppp_mu
    mu_list[3,i]=pppp_mu
    mu_list[4,i]=ppppp_mu


    p_bc=generate_boundarycharge(n=N,z=Z,dt=DT,u=U,du=DU,v=V,filling=f,
                                  mu=mu_list[0,i],phi_v=phi,phi_t=phi)
    pp_bc=generate_boundarycharge(n=N,z=Z,dt=DT*pp,u=U,du=DU,v=V*pp,filling=f,
                                  mu=mu_list[1,i],phi_v=phi,phi_t=phi)
    ppp_bc=generate_boundarycharge(n=N,z=Z,dt=DT*ppp,u=U,du=DU,v=V*ppp,filling=f,
                                  mu=mu_list[2,i],phi_v=phi,phi_t=phi)
    pppp_bc=generate_boundarycharge(n=N,z=Z,dt=DT*pppp,u=U,du=DU,v=V*pppp,filling=f,
                                  mu=mu_list[3,i],phi_v=phi,phi_t=phi)
    ppppp_bc=generate_boundarycharge(n=N,z=Z,dt=DT*ppppp,u=U,du=DU,v=V*ppppp,filling=f,
                                  mu=mu_list[4,i],phi_v=phi,phi_t=phi)

    qb_list[0,i]=p_bc 
    qb_list[1,i]=pp_bc
    qb_list[2,i]=ppp_bc   
    qb_list[3,i]=pppp_bc             
    qb_list[4,i]=ppppp_bc


    print('==========')
    
np.save('frg_data/bc_3sqrt_varing_dtandV.npy',qb_list)
np.save('frg_data/bc_3sqrt_varing_dtandV_mu.npy',mu_list)
