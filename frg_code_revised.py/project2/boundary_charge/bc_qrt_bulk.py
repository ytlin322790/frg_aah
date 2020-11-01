import numpy as np
import sys
sys.path.append('../..')
from tool.frg_aah_tool import dQdgamma,eff_boundary_charge_z4
sys.path.append('project1_modulation_of_interaction_z=4/boudnary_charge')
from parameter.parameters_bc_qrt_filling import *


mu_list_a=np.load('data/mu_qrt_dU=0.025.npy')
mu_list_b=np.load('data/mu_qrt_dU=0.05.npy')
mu_list_c=np.load('data/mu_qrt_dU=0.075.npy')
mu_list_d=np.load('data/mu_qrt_dU=0.125.npy')
mu_list_e=np.load('data/mu_qrt_dU=0.25.npy')


bc_list_a=np.zeros(NNN,dtype=np.float64)
bc_list_b=np.zeros(NNN,dtype=np.float64)
bc_list_c=np.zeros(NNN,dtype=np.float64)
bc_list_d=np.zeros(NNN,dtype=np.float64)
bc_list_e=np.zeros(NNN,dtype=np.float64)

for i in range(NNN):
    phi=2.0*np.pi*i/NNN+0.001

    bc_a=eff_boundary_charge_z4(n=N,z=Z,dt=DT,u=U,du=DU*P1,v=V,mu=mu_list_a[i],
                                filling=FILLING,phi_u=phi)
    bc_b=eff_boundary_charge_z4(n=N,z=Z,dt=DT,u=U,du=DU*P2,v=V,mu=mu_list_b[i],
                                filling=FILLING,phi_u=phi)
    bc_c=eff_boundary_charge_z4(n=N,z=Z,dt=DT,u=U,du=DU*P3,v=V,mu=mu_list_c[i],
                                filling=FILLING,phi_u=phi)
    bc_d=eff_boundary_charge_z4(n=N,z=Z,dt=DT,u=U,du=DU*P4,v=V,mu=mu_list_d[i],
                                filling=FILLING,phi_u=phi)
    bc_e=eff_boundary_charge_z4(n=N,z=Z,dt=DT,u=U,du=DU*P5,v=V,mu=mu_list_e[i],
                                filling=FILLING,phi_u=phi)
    
 
    bc_list_a[i]=bc_a
    bc_list_b[i]=bc_b
    bc_list_c[i]=bc_c
    bc_list_d[i]=bc_d
    bc_list_e[i]=bc_e
    print(bc_a,bc_b,bc_c,bc_d,bc_e)
    print('==========')
    print(i)

phi_list=np.zeros(NNN,dtype=np.float64)
for i in range(NNN):
    phi=2.0*np.pi*i/NNN+0.001
    phi_list[i]=phi/(2.0*np.pi)

A=dQdgamma(phi_list,bc_list_a)
B=dQdgamma(phi_list,bc_list_b)
C=dQdgamma(phi_list,bc_list_c)
D=dQdgamma(phi_list,bc_list_d)
E=dQdgamma(phi_list,bc_list_e)

   
