import numpy as np
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import sys
sys.path.append('../..')
from numpy import exp

from frg.tool import dQdgamma,eff_boundary_charge_z4,envelope_f

sys.path.append('project1')
from parameter.parameters_bc_3qrt_filling import *

qb_green_list_1=np.load('data/bc_3qrt_v=0.015_dt=0.02.npy')
qb_green_list_2=np.load('data/bc_3qrt_v=0.075_dt=0.1.npy')
qb_green_list_3=np.load('data/bc_3qrt_v=0.15_dt=0.2.npy')
qb_green_list_4=np.load('data/bc_3qrt_v=0.75_dt=1.npy')
qb_green_list_5=np.load('data/bc_3qrt_v=1.5_dt=2.npy')
qb_green_list_6=np.load('data/bc_3qrt_v=15_dt=20.npy')


qb_vertex_list_1=np.load('data/QB_v=0.015_dt=0.02.npy')
qb_vertex_list_2=np.load('data/QB_v=0.075_dt=0.1.npy')
qb_vertex_list_3=np.load('data/QB_v=0.15_dt=0.2.npy')
qb_vertex_list_4=np.load('data/QB_v=0.75_dt=1.npy')
qb_vertex_list_5=np.load('data/QB_v=1.5_dt=2.npy')

qb_bulk_list_1=np.load('data/bc_bulk_3qrt_v=0.015_dt=0.02.npy')
qb_bulk_list_2=np.load('data/bc_bulk_3qrt_v=0.075_dt=0.1.npy')
qb_bulk_list_3=np.load('data/bc_bulk_3qrt_v=0.15_dt=0.2.npy')
qb_bulk_list_4=np.load('data/bc_bulk_3qrt_v=0.75_dt=1.npy')
qb_bulk_list_5=np.load('data/bc_bulk_3qrt_v=1.5_dt=2.npy')


NNN=41

phi_list=np.zeros(NNN,dtype=np.float64)
for i in range(NNN):
    phi=2.0*np.pi*i/NNN+0.001
    phi_list[i]=phi/(2.0*np.pi)


LD_vertex_a=np.load('data/vertex_3qrt/LD_v=0.015_dt=0.02.npy')
LD_vertex_b=np.load('data/vertex_3qrt//LD_v=0.075_dt=0.1.npy')
LD_vertex_c=np.load('data/vertex_3qrt//LD_v=0.15_dt=0.2.npy')
LD_vertex_d=np.load('data/vertex_3qrt//LD_v=0.75_dt=1.npy')
LD_vertex_e=np.load('data/vertex_3qrt//LD_v=1.5_dt=2.npy')

bc_list_vertex_a=np.zeros(NNN,dtype=np.float64)
bc_list_vertex_b=np.zeros(NNN,dtype=np.float64)
bc_list_vertex_c=np.zeros(NNN,dtype=np.float64)
bc_list_vertex_d=np.zeros(NNN,dtype=np.float64)
bc_list_vertex_e=np.zeros(NNN,dtype=np.float64)



for i in range(NNN):
    f1=envelope_f(N,k1=0.5,k2=15)
    bc_list_vertex_a[i]=np.sum((LD_vertex_a[i,0:np.int(N/2):1]-np.average(LD_vertex_a[i,np.int(N/2)-200:np.int(N/2):1]))*f1)
    bc_list_vertex_b[i]=np.sum((LD_vertex_b[i,0:np.int(N/2):1]-np.average(LD_vertex_b[i,np.int(N/2)-200:np.int(N/2):1]))*f1)
    bc_list_vertex_c[i]=np.sum((LD_vertex_c[i,0:np.int(N/2):1]-np.average(LD_vertex_c[i,np.int(N/2)-200:np.int(N/2):1]))*f1)
    bc_list_vertex_d[i]=np.sum((LD_vertex_d[i,0:np.int(N/2):1]-np.average(LD_vertex_d[i,np.int(N/2)-200:np.int(N/2):1]))*f1)
    bc_list_vertex_e[i]=np.sum((LD_vertex_e[i,0:np.int(N/2):1]-np.average(LD_vertex_e[i,np.int(N/2)-200:np.int(N/2):1]))*f1)
    


A=dQdgamma(phi_list,bc_list_vertex_a)
B=dQdgamma(phi_list,bc_list_vertex_b)
C=dQdgamma(phi_list,bc_list_vertex_c)
D=dQdgamma(phi_list,bc_list_vertex_d)
E=dQdgamma(phi_list,bc_list_vertex_e)

AA=dQdgamma(phi_list,qb_green_list_1)
BB=dQdgamma(phi_list,qb_green_list_2)
CC=dQdgamma(phi_list,qb_green_list_3)
DD=dQdgamma(phi_list,qb_green_list_4)
EE=dQdgamma(phi_list,qb_green_list_5)

AAA=dQdgamma(phi_list,qb_bulk_list_1)
BBB=dQdgamma(phi_list,qb_bulk_list_2)
CCC=dQdgamma(phi_list,qb_bulk_list_3)
DDD=dQdgamma(phi_list,qb_bulk_list_4)
EEE=dQdgamma(phi_list,qb_bulk_list_5)

C_0=np.delete(C[0],8)
C_1=np.delete(C[1],8)
D_0=np.delete(D[0],8)
D_1=np.delete(D[1],8)
E_0=np.delete(E[0],8)
E_1=np.delete(E[1],8)

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



fig,ax=plt.subplots()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.plot(phi_list,bc_list_vertex_a,'.',label=r"$V$=0.015,$\delta t$=0.02",zorder=1300,markersize=7,rasterized=True)#
#plt.plot(phi_list,bc_list_vertex_b,'o',label=r"$V$=0.075,$\Delta t$=0.1",rasterized=True)#
#plt.plot(phi_list,bc_list_vertex_c,'o',label=r"$V$=0.15,$\Delta t$=0.2",zorder=1200,rasterized=True)#
plt.plot(phi_list,bc_list_vertex_d,'o',label=r"$V$=0.75,$\delta t$=1",zorder=1100,rasterized=True)#
plt.plot(phi_list,bc_list_vertex_e,'^',label=r"$V$=1.5,$\delta t$=2",zorder=1000,rasterized=True)#
plt.plot(phi_list,qb_green_list_6,'D',label=r"$V$=15,$\delta t$=20",zorder=1000,rasterized=True)#

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


plt.ylim([-1.25,-0.75])
plt.xlim([-0.05,1.05])
plt.plot(A[0],A[1],'.'  ,zorder=1000,markersize=4,rasterized=True)
plt.plot(D_0,D_1,'o'  ,zorder=1100 ,markersize=4,rasterized=True)
plt.plot(E_0,E_1,'^'  ,zorder=1200 ,markersize=3,rasterized=True)

plt.plot(AAA_0,AAA_1,':',color='k' ,zorder=1300,linewidth=0.8,markersize=1.5,rasterized=True)
plt.plot(CCC_0,CCC_1,'-',color='k' ,zorder=1300,linewidth=0.8 ,markersize=1.5,rasterized=True)
plt.plot(DDD_0,DDD_1,'-',color='k' ,zorder=1300,linewidth=0.8 ,markersize=1.5,rasterized=True)
plt.plot(EEE_0,EEE_1,'-',color='k' ,zorder=1300,linewidth=0.8 ,markersize=1.5,rasterized=True)
plt.ylabel(r'$2\pi\frac{d}{d\gamma}Q_B$',fontsize=14,labelpad=0.5)
plt.savefig('plot/bc_aahmodel_3qrtfilling_vertex.pdf',format='pdf',dpi=300)
plt.show() 


fig,ax=plt.subplots()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot(phi_list,qb_green_list_1,'*',label=r"V=0.015,$\Delta t$=0.02",rasterized=True)#
plt.plot(phi_list,qb_green_list_2,'*',label=r"V=0.075,$\Delta t$=0.1",rasterized=True)#
plt.plot(phi_list,qb_green_list_3,'*',label=r"V=0.15,$\Delta t$=0.2",rasterized=True)#
plt.plot(phi_list,qb_green_list_4,'*',label=r"V=0.75,$\Delta t$=1",rasterized=True)#
plt.plot(phi_list,qb_green_list_5,'*',label=r"V=1.5,$\Delta t$=2",rasterized=True)#

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

plt.ylim([-1.25,-0.75])

plt.xlim([-0.05,1.05])
plt.plot(AA[0],AA[1],'.'  ,zorder=1100,markersize=3,rasterized=True)
plt.plot(BB[0],BB[1],'.'  ,zorder=700 ,markersize=3,rasterized=True)
plt.plot(CC[0],CC[1],'^'  ,zorder=800 ,markersize=3,rasterized=True)
plt.plot(DD[0],DD[1],'o--'  ,zorder=800 ,markersize=3,rasterized=True)
plt.plot(EE[0],EE[1],'D--'  ,zorder=800 ,markersize=3,rasterized=True)

plt.ylabel(r'$2\pi\frac{d}{d\gamma}Q_B$',fontsize=14,labelpad=0.5)


plt.tight_layout()
plt.savefig('plot/bc_aahmodel_3sqrtfilling_green.pdf',format='pdf',dpi=300)
plt.show() 

