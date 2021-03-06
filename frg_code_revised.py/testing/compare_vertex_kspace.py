import sys 
import numpy as np



Z=4
NN=20
DT=-0.005
V=0.013
MU=0.0
U=0.125
DU=0.0


N=40000
TARGET=10000
z=Z
dt=DT
v=V
mu=MU
u=U
du=DU
n=N
PHI_V=PHI_T=PHI_U=0.0


site_a=0
site_b=1
site_c=2
site_d=3

a=b=c=d=e=0.0

if __name__=='__main__':
    sys.path.append('../..')
    from frg.tool import search_mu_green
    from frg.greenfunction import FrgGreenData
    from frg.parallel import kspace_vertex_aah_z4_parallel
    
    
    mu_frg=search_mu_green(n=N,z=Z, 
                           dt=DT,u=U,du=DU,v=V, 
                           pn=TARGET,mu_upper=-1.5,mu_lower=-2.0,
                           phi_v=PHI_V,phi_t=PHI_T,phi_u=PHI_U)
    
    kspace_list=np.zeros(NN,dtype=np.float64)
    greenfunction_list=np.zeros(NN,dtype=np.float64)
    for i in range(NN):
        MU=mu_frg+0.0001*NN
        sol=kspace_vertex_aah_z4_parallel(Z,DT,U,V,DU,PHI_V,PHI_T,PHI_U,MU)
        print(sol)
        pd=np.sum(sol)/Z
        kspace_list[i]=pd
        print(pd)

    np.save('kspace_particle_density_v=%g,dt=%g.npy'%(V,DT),kspace_list)


    for i in range(NN):
        MU=mu_frg+0.0001*NN
        k=FrgGreenData()
        k.frg_obc(n=N,z=Z,dt=DT,u=U,v=V,du=DU,mu=MU)
        k.local_density(n=N,z=Z,dt=DT,u=U,v=V,du=DU,mu=MU)
        rho=k.get_local_density()  
        ans=(rho[site_a+20000]+rho[site_b+20000]+rho[site_c+20000]+rho[site_d+20000])/Z
        print(ans)
        greenfunction_list[i]=ans
    np.save('gf_particle_density_v=%g,dt=%g.npy'%(V,DT),greenfunction_list)

        

