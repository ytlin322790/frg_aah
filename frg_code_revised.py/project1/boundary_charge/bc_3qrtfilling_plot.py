import numpy as np
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import sys
sys.path.append('../..')


from frg.tool import dQdgamma

sys.path.append('project1/boundary_charge')
from parameter.parameters_bc_3qrt_filling import *

qb_list=np.load('data/bc_3sqrt_varing_dtandV.npy')
qb_list_mu=np.load('data/bc_3sqrt_varing_dtandV_mu.npy')


NNN=41

phi_list=np.zeros(NNN,dtype=np.float64)
for i in range(NNN):
    phi=2.0*np.pi*i/NNN+0.001
    phi_list[i]=phi/(2.0*np.pi)



A=dQdgamma(phi_list,qb_list[0,:])
B=dQdgamma(phi_list,qb_list[1,:])
C=dQdgamma(phi_list,qb_list[2,:])

fig,ax=plt.subplots()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot(phi_list,qb_list[0,:],'.',label=r"V=2.5$\Delta t$=0.05",rasterized=True)#
plt.plot(phi_list,qb_list[1,:],'o',label=r"V=2.5$\Delta t$=0.25",rasterized=True)#
plt.plot(phi_list,qb_list[2,:],'d',label=r"V=2.5$\Delta t$=0.5",rasterized=True)#
plt.plot(phi_list,qb_list[3,:],'^',label=r"V=2.5$\Delta t$=2.5",rasterized=True)#
plt.plot(phi_list,qb_list[4,:],'^',label=r"V=2.5$\Delta t$=5.0",rasterized=True)#


plt.legend(loc='best')
plt.ylim([-0.55,0.55])
my_x_ticks = [0.0,0.25,0.5,0.75,1.0]
plt.xticks(my_x_ticks)
my_y_ticks = [-0.5,-0.25,0.0,0.25,0.5]
plt.yticks(my_y_ticks)
plt.xlabel(r'$\gamma/2\pi $',fontsize=18)
plt.ylabel(r'$Q_B$',fontsize=18)
plt.tight_layout()
plt.savefig('plot/bc_aahmodel_3sqrtfilling_new.pdf',format='pdf',dpi=300)
plt.show() 

print(qb_list_mu)
