import numpy as np
import sys
sys.path.append('../..')
from tool.tool import dQdgamma,eff_boundary_charge_z4
from parameter.parameters_bc_3qrt_filling import *
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt


sys.path.append('project2_interacting_aah_model_z=4')
mu_list_a=np.load('data/mu_3qrt_v=0.015_dt=0.02.npy')
mu_list_b=np.load('data/mu_3qrt_v=0.075_dt=0.1.npy')
mu_list_c=np.load('data/mu_3qrt_v=0.15_dt=0.2.npy')
mu_list_d=np.load('data/mu_3qrt_v=0.75_dt=1.npy')
mu_list_e=np.load('data/mu_3qrt_v=1.5_dt=2.npy')


bc_list_a=np.zeros(NNN,dtype=np.float64)
bc_list_b=np.zeros(NNN,dtype=np.float64)
bc_list_c=np.zeros(NNN,dtype=np.float64)
bc_list_d=np.zeros(NNN,dtype=np.float64)
bc_list_e=np.zeros(NNN,dtype=np.float64)



for i in range(NNN):
    phi=2.0*np.pi*i/NNN+0.001

    bc_a=eff_boundary_charge_z4(n=N,z=Z,dt=DT*P1,u=U,du=DU,v=V*P1,mu=mu_list_a[i],
                                filling=FILLING,phi_v=phi,phi_t=phi)
    bc_b=eff_boundary_charge_z4(n=N,z=Z,dt=DT*P2,u=U,du=DU,v=V*P2,mu=mu_list_b[i],
                                filling=FILLING,phi_v=phi,phi_t=phi)
    bc_c=eff_boundary_charge_z4(n=N,z=Z,dt=DT*P3,u=U,du=DU,v=V*P3,mu=mu_list_c[i],
                                filling=FILLING,phi_v=phi,phi_t=phi)
    bc_d=eff_boundary_charge_z4(n=N,z=Z,dt=DT*P4,u=U,du=DU,v=V*P4,mu=mu_list_d[i],
                                filling=FILLING,phi_v=phi,phi_t=phi)
    bc_e=eff_boundary_charge_z4(n=N,z=Z,dt=DT*P5,u=U,du=DU,v=V*P5,mu=mu_list_e[i],
                                filling=FILLING,phi_v=phi,phi_t=phi)
    
 
    bc_list_a[i]=bc_a
    bc_list_b[i]=bc_b
    bc_list_c[i]=bc_c
    bc_list_d[i]=bc_d
    bc_list_e[i]=bc_e

    print('==========')


np.save('data/bc_bulk_3qrt_v=0.015_dt=0.02.npy',bc_list_a)
np.save('data/bc_bulk_3qrt_v=0.075_dt=0.1.npy',bc_list_b)
np.save('data/bc_bulk_3qrt_v=0.15_dt=0.2.npy',bc_list_c)
np.save('data/bc_bulk_3qrt_v=0.75_dt=1.npy',bc_list_d)
np.save('data/bc_bulk_3qrt_v=1.5_dt=2.npy',bc_list_e)






phi_list=np.zeros(NNN,dtype=np.float64)
for i in range(NNN):
    phi=2.0*np.pi*i/NNN+0.001
    phi_list[i]=phi/(2.0*np.pi)
 
AAA=dQdgamma(phi_list,bc_list_a)
BBB=dQdgamma(phi_list,bc_list_b)
CCC=dQdgamma(phi_list,bc_list_c)
DDD=dQdgamma(phi_list,bc_list_d)
EEE=dQdgamma(phi_list,bc_list_e)


fig,ax=plt.subplots()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.plot(phi_list,bc_list_a,'*',label=r"V=0.015,$\Delta t$=0.02",rasterized=True)#
plt.plot(phi_list,bc_list_b,'*',label=r"V=0.075,$\Delta t$=0.1",rasterized=True)#
plt.plot(phi_list,bc_list_c,'*',label=r"V=0.15,$\Delta t$=0.2",rasterized=True)#
plt.plot(phi_list,bc_list_d,'*',label=r"V=0.75,$\Delta t$=1",rasterized=True)#
plt.plot(phi_list,bc_list_e,'*',label=r"V=1.5,$\Delta t$=2",rasterized=True)#

plt.legend(loc='upper right')
plt.ylim([-0.53,0.53])
my_x_ticks = [0.0,0.25,0.5,0.75,1.0]
plt.xticks(my_x_ticks)
my_y_ticks = [-0.5,-0.25,0.0,0.25,0.5]
plt.yticks(my_y_ticks)
plt.xlabel(r'$\gamma/2\pi $',fontsize=18)
plt.ylabel(r'$Q_B$',fontsize=18)
plt.tight_layout()
ax2 = fig.add_axes([0.35, 0.4, 0.25, 0.25])
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)


ax2.xaxis.set_minor_locator(MultipleLocator(0.25))

AAA_0=np.delete(AAA[0],8)
AAA_1=np.delete(AAA[1],8)
BBB_0=np.delete(BBB[0],8)
BBB_1=np.delete(BBB[1],8)
CCC_0=np.delete(CCC[0],8)
CCC_1=np.delete(CCC[1],8)
DDD_0=np.delete(DDD[0],8)
DDD_1=np.delete(DDD[1],8)
EEE_0=np.delete(EEE[0],8)
EEE_1=np.delete(EEE[1],8)
plt.ylim([-1.3,-0.7])
plt.xlim([-0.05,1.05])
plt.plot(AAA_0,AAA_1,'.'  ,zorder=1100,markersize=3,rasterized=True)
plt.plot(BBB_0,BBB_1,'.'  ,zorder=700 ,markersize=3,rasterized=True)
plt.plot(CCC_0,CCC_1,'^'  ,zorder=800 ,markersize=3,rasterized=True)
plt.plot(DDD_0,DDD_1,'o--'  ,zorder=800 ,markersize=3,rasterized=True)
plt.plot(EEE_0,EEE_1,'D--'  ,zorder=800 ,markersize=3,rasterized=True)
plt.ylabel(r'$2\pi\frac{d}{d\gamma}Q_B$',fontsize=14,labelpad=0.5)
plt.tight_layout()
plt.savefig('plot/bc_aahmodel_3qrtfilling_effbulk.pdf',format='pdf',dpi=300)
plt.show() 
