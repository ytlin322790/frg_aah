import numpy as np
from numpy import pi
import sys 
sys.path.append('../..')

from frg.greenfunction import FrgGreenData
from frg.tool import search_mu_green,dQdgamma
from parameters import *
from sqrt_filling_mu_list import *


d=5
mu_list=np.zeros([5,21],dtype=np.float64)
qb_list=np.zeros([5,21],dtype=np.float64)
phi_list=np.zeros(21,dtype=np.float64)

mu_list[0,:]=MU_LIST_U005_dU01
mu_list[1,:]=MU_LIST_U005_dU1
mu_list[2,:]=MU_LIST_U005_dU10
mu_list[3,:]=MU_LIST_U005_dU100
mu_list[4,:]=MU_LIST_U005_dU1000

for i in range(21):

    phi=2.0*np.pi*i/20+0.001

    phi_list[i]=phi/(2.0*np.pi)
    U=U_0
    DU_1=DU_0*1.0
    DU_10=DU_0*10.0
    DU_100=DU_0*100.0
    DU_1000=DU_0*1000.0
    DU_10000=DU_0*10000.0

    mu_1=mu_list[0,i]
    mu_10=mu_list[1,i]
    mu_100=mu_list[2,i]
    mu_1000=mu_list[3,i]
    mu_10000=mu_list[4,i]
    
    
    p1=FrgGreenData()
    p1.frg_obc(n=N,z=Z,dt=DT,u=U,v=V,du=DU_1,mu=mu_1,phi_u=phi)
    p1.local_density(n=N,z=Z,dt=DT,u=U,v=V,du=DU_1,mu=mu_1,phi_u=phi)
    rho=p1.get_local_density()

    print('particle number:',np.around(np.sum(rho),d))
    print('final mu:',i,mu_1)

    p1.boundary_charge(n=N,filling=0.25)
    qb_1=p1.get_boundary_charge()

    qb_list[0,i]=qb_1
    print('-----------')

    p2=FrgGreenData()
    p2.frg_obc(n=N,z=Z,dt=DT,u=U,v=V,du=DU_10,mu=mu_10,phi_u=phi)
    p2.local_density(n=N,z=Z,dt=DT,u=U,v=V,du=DU_10,mu=mu_10,phi_u=phi)
    rho=p2.get_local_density()

    print('particle number:',np.around(np.sum(rho),d))
    print('final mu:',i,mu_10)

    p2.boundary_charge(n=N,filling=0.25)
    qb_2=p2.get_boundary_charge()

    qb_list[1,i]=qb_2
    print('-----------')
    

    p3=FrgGreenData()
    p3.frg_obc(n=N,z=Z,dt=DT,u=U,v=V,du=DU_100,mu=mu_100,phi_u=phi)
    p3.local_density(n=N,z=Z,dt=DT,u=U,v=V,du=DU_100,mu=mu_100,phi_u=phi)
    rho=p3.get_local_density()

    print('particle number:',np.around(np.sum(rho),d))
    print('final mu:',i,mu_100)

    p3.boundary_charge(n=N,filling=0.25)
    qb_3=p3.get_boundary_charge()

    qb_list[2,i]=qb_3
    print('-----------')

    p4=FrgGreenData()
    p4.frg_obc(n=N,z=Z,dt=DT,u=U,v=V,du=DU_1000,mu=mu_1000,phi_u=phi)
    p4.local_density(n=N,z=Z,dt=DT,u=U,v=V,du=DU_1000,mu=mu_1000,phi_u=phi)
    rho=p4.get_local_density()

    print('particle number:',np.around(np.sum(rho),d))
    print('final mu:',i,mu_1000)

    p4.boundary_charge(n=N,filling=0.25)
    qb_4=p4.get_boundary_charge()

    qb_list[3,i]=qb_4
    print('-----------')

    p5=FrgGreenData()
    p5.frg_obc(n=N,z=Z,dt=DT,u=U,v=V,du=DU_10000,mu=mu_10000,phi_u=phi)
    p5.local_density(n=N,z=Z,dt=DT,u=U,v=V,du=DU_10000,mu=mu_10000,phi_u=phi)
    rho=p5.get_local_density()

    print('particle number:',np.around(np.sum(rho),d))
    print('final mu:',i,mu_10000)

    p5.boundary_charge(n=N,filling=0.25)
    qb_5=p5.get_boundary_charge()

    qb_list[4,i]=qb_5
    print('==========')


A=dQdgamma(phi_list,qb_list[0,:])
B=dQdgamma(phi_list,qb_list[1,:])
C=dQdgamma(phi_list,qb_list[2,:])
D=dQdgamma(phi_list,qb_list[3,:])
EE=dQdgamma(phi_list,qb_list[4,:])

