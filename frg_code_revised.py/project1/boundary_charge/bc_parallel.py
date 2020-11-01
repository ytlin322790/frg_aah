import numpy as np
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import sys 

a=b=c=d=e=0.0

if __name__ == '__main__':
    sys.path.append('../..')

    from frg.tool import generate_boundarycharge
    from frg.parallel import find_mu_aah_z4_parallel

    sys.path.append('/project1')

    from parameter.parameters_bc_3qrt_filling import *

    a=find_mu_aah_z4_parallel(n_phi=NNN,n=N,z=Z,dt=DT*P1,u=U,v=V*P1,du=DU,pn=TARGET,mu_upper=MU_UPPER_1,mu_lower=MU_LOWER_1)
    b=find_mu_aah_z4_parallel(n_phi=NNN,n=N,z=Z,dt=DT*P2,u=U,v=V*P2,du=DU,pn=TARGET,mu_upper=MU_UPPER_2,mu_lower=MU_LOWER_2)
    c=find_mu_aah_z4_parallel(n_phi=NNN,n=N,z=Z,dt=DT*P3,u=U,v=V*P3,du=DU,pn=TARGET,mu_upper=MU_UPPER_3,mu_lower=MU_LOWER_3)
    d=find_mu_aah_z4_parallel(n_phi=NNN,n=N,z=Z,dt=DT*P4,u=U,v=V*P4,du=DU,pn=TARGET,mu_upper=MU_UPPER_4,mu_lower=MU_LOWER_4)
    e=find_mu_aah_z4_parallel(n_phi=NNN,n=N,z=Z,dt=DT*P5,u=U,v=V*P5,du=DU,pn=TARGET,mu_upper=MU_UPPER_5,mu_lower=MU_LOWER_5)

    mu_list_a=a
    mu_list_b=b
    mu_list_c=c
    mu_list_d=d
    mu_list_e=e

    np.save('data/mu_3qrt_v=%g_dt=%g.npy'%(V*P1,DT*P1),mu_list_a)
    np.save('data/mu_3qrt_v=%g_dt=%g.npy'%(P2*V,P2*DT),mu_list_b)
    np.save('data/mu_3qrt_v=%g_dt=%g.npy'%(P3*V,P3*DT),mu_list_c)
    np.save('data/mu_3qrt_v=%g_dt=%g.npy'%(P4*V,P4*DT),mu_list_d)
    np.save('data/mu_3qrt_v=%g_dt=%g.npy'%(P5*V,P5*DT),mu_list_e)



    
    bc_list_a=np.zeros(NNN,dtype=np.float64)
    bc_list_b=np.zeros(NNN,dtype=np.float64)
    bc_list_c=np.zeros(NNN,dtype=np.float64)
    bc_list_d=np.zeros(NNN,dtype=np.float64)
    bc_list_e=np.zeros(NNN,dtype=np.float64)




    for i in range(NNN):
        phi=2.0*np.pi*i/NNN+0.001

        bc_a=generate_boundarycharge(n=N,z=Z,dt=DT*P1,u=U,du=DU,v=V*P1,filling=FILLING,
                                     mu=mu_list_a[i],phi_v=phi,phi_t=phi)
        bc_b=generate_boundarycharge(n=N,z=Z,dt=DT*P2,u=U,du=DU,v=V*P2,filling=FILLING,
                                     mu=mu_list_b[i],phi_v=phi,phi_t=phi)
        bc_c=generate_boundarycharge(n=N,z=Z,dt=DT*P3,u=U,du=DU,v=V*P3,filling=FILLING,
                                     mu=mu_list_c[i],phi_v=phi,phi_t=phi)
        bc_d=generate_boundarycharge(n=N,z=Z,dt=DT*P4,u=U,du=DU,v=V*P4,filling=FILLING,
                                     mu=mu_list_d[i],phi_v=phi,phi_t=phi)
        bc_e=generate_boundarycharge(n=N,z=Z,dt=DT*P5,u=U,du=DU,v=V*P5,filling=FILLING,
                                     mu=mu_list_e[i],phi_v=phi,phi_t=phi)
    
 
        bc_list_a[i]=bc_a
        bc_list_b[i]=bc_b
        bc_list_c[i]=bc_c
        bc_list_d[i]=bc_d
        bc_list_e[i]=bc_e
        print('==========')


    np.save('data/mu_3qrt_v=%g_dt=%g.npy'%(V*P1,DT*P1),mu_list_a)
    np.save('data/bc_3qrt_v=%g_dt=%g.npy'%(V*P1,DT*P1),bc_list_a)

    np.save('data/mu_3qrt_v=%g_dt=%g.npy'%(P2*V,P2*DT),mu_list_b)
    np.save('data/bc_3qrt_v=%g_dt=%g.npy'%(P2*V,P2*DT),bc_list_b)


    np.save('data/mu_3qrt_v=%g_dt=%g.npy'%(P3*V,P3*DT),mu_list_c)
    np.save('data/bc_3qrt_v=%g_dt=%g.npy'%(P3*V,P3*DT),bc_list_c)


    np.save('data/mu_3qrt_v=%g_dt=%g.npy'%(P4*V,P4*DT),mu_list_d)
    np.save('data/bc_3qrt_v=%g_dt=%g.npy'%(P4*V,P4*DT),bc_list_d)


    np.save('data/mu_3qrt_v=%g_dt=%g.npy'%(P5*V,P5*DT),mu_list_e)
    np.save('data/bc_3qrt_v=%g_dt=%g.npy'%(P5*V,P5*DT),bc_list_e)

