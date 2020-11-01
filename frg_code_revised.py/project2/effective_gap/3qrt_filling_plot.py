import numpy as np
from numpy import sin,cos,arccos
import sys 
import matplotlib.pyplot as plt
sys.path.append('../..')
from frg.derivative import log_log_der

sys.path.append('project2/effective_gap')
from parameter.three_qrt_filling import N_GAP,N,Z,DT,V,DU_0,PHI_V,PHI_T,PHI_U,U_0,N_U,TARGET
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as ticker
import matplotlib

print('U:',U_0)
print('dU:',DU_0)
print('v:',V)
print('dt:',DT)

frg_bulk_gap=np.load('data/frg_gap_3qrt_dU=%g_U=%g.npy'%(DU_0,U_0))
free_bulk_gap=np.load('data/free_gap_3qrt_dU=%g_U=%g.npy'%(DU_0,U_0))
free_mu=np.load('data/mu_free_gap_3qrt_dU=%g_U=%g.npy'%(DU_0,U_0))
frg_mu=np.load('data/mu_gap_3qrt_dU=%g_U=%g.npy'%(DU_0,U_0))

u_ref=np.zeros(N_U,dtype=np.float64)
u_data=np.zeros(N_U,dtype=np.float64)
gap1_data=np.zeros([N_U,N_GAP],dtype=np.float64)
gap2_data=np.zeros([N_U,N_GAP],dtype=np.float64)
gap3_data=np.zeros([N_U,N_GAP],dtype=np.float64)

exponent1_data=np.zeros([N_U,N_GAP-2,2],dtype=np.float64)
exponent2_data=np.zeros([N_U,N_GAP-2,2],dtype=np.float64)
exponent3_data=np.zeros([N_U,N_GAP-2,2],dtype=np.float64)

beta=np.zeros(N_U,dtype=np.float64)

x_list=np.zeros(N_U,dtype=np.float64)
for ii in range(N_GAP):

    delta=0.005*ii
    DU=DU_0+delta
    x_list[ii]=DU
    
for i in range(N_U):
    U=U_0*i
    u_data[i]=U
    
    mu_0=free_mu[i][0]
    kF=arccos(-mu_0/2.0)
    b=-U*(1.0-cos(2.0*kF))/(2.0*np.pi*sin(kF))

    beta[i]=b
    u_ref[i]=-U*(1.0-cos(2.0*kF))/(2.0*np.pi*sin(kF))

    AA=log_log_der(free_bulk_gap[i,:,0],np.abs(frg_bulk_gap[i,:,0]/free_bulk_gap[i,:,0]))

    CC=log_log_der(free_bulk_gap[i,:,2],np.abs(frg_bulk_gap[i,:,2]/free_bulk_gap[i,:,2]))


    gap1_data[i,:]=np.abs(frg_bulk_gap[i,:,0]/free_bulk_gap[i,:,0])

    gap3_data[i,:]=np.abs(frg_bulk_gap[i,:,2]/free_bulk_gap[i,:,2])
    
    exponent1_data[i,:,0]=AA[0]
    exponent1_data[i,:,1]=AA[1]

    exponent3_data[i,:,0]=CC[0]
    exponent3_data[i,:,1]=CC[1]




fig,ax=plt.subplots()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


ax.set_xscale('log')
ax.set_yscale('log')


ax.xaxis.set_minor_locator(MultipleLocator(10))
ax.xaxis.set_ticklabels([])
ax.yaxis.set_minor_locator(MultipleLocator(10))
ax.yaxis.set_ticklabels([])
plt.xticks([0.01,0.02,0.04])
plt.yticks([1.0, 1.3, 1.6])
plt.plot(free_bulk_gap[0,:,2],gap3_data[0,:],'D-' )
plt.plot(free_bulk_gap[2,:,2],gap3_data[2,:],'D-' )
plt.plot(free_bulk_gap[4,:,2],gap3_data[4,:],'D-' )
plt.plot(free_bulk_gap[9,:,2],gap3_data[9,:],'D-')


plt.plot([0.025,0.029]  , [1.64,1.64]  ,'--'   ,markerfacecolor='k',color='k',rasterized=True)

ax.text(0.03, 1.62  ,r'$-\frac{U (1-2\cos(2 k_F))}{2\pi  \sin(k_F)}$'  ,fontsize=13.)

ax.text(0.0067, 0.99,'$1$'  ,fontsize=13.5)
ax.text(0.0064, 1.29,'$1.3$',fontsize=13.5)
ax.text(0.0064, 1.59,'$1.6$',fontsize=13.5)

ax.text(0.0097, 0.92,'$1$'  ,fontsize=13.5)
ax.text(0.0195, 0.92,'$2$'  ,fontsize=13.5)
ax.text(0.039 , 0.92,'$4$'  ,fontsize=13.5)
ax.text(0.043 , 0.93,'$10^{-2}$'  ,fontsize=10.5)

plt.ylim([0.98,1.7])

plt.xlabel(r'$2\Delta^{U=0}_{\nu}$',fontsize=18,labelpad=17)
plt.ylabel(r'$2\Delta_{\nu} / 2\Delta^{U=0}_{\nu}$',fontsize=18,labelpad=19)
plt.tight_layout()

ax2 = fig.add_axes([0.28, 0.62, 0.23, 0.2])

plt.xticks(fontsize=10.6)
plt.yticks(fontsize=11)
plt.ylim([-0.115,0.005])
ax.text(0.037, 1.02,'$U=0$'   ,fontsize=12.)
ax.text(0.037, 1.12,'$U=0.1$' ,fontsize=12.)
ax.text(0.037, 1.24,'$U=0.2$' ,fontsize=12.)
ax.text(0.035, 1.37,'$U=0.45$' ,fontsize=12.)

plt.plot(exponent3_data[0,:,0],exponent3_data[0,:,1],'D-',markersize=4, rasterized=True)#
plt.plot(exponent3_data[2,:,0],exponent3_data[2,:,1],'D-',markersize=4,rasterized=True)#
plt.plot(exponent3_data[4,:,0],exponent3_data[4,:,1],'D-',markersize=4,rasterized=True)#
plt.plot(exponent3_data[9,:,0],exponent3_data[9,:,1],'D-',markersize=4,rasterized=True)#




plt.hlines(beta[0], exponent3_data[0,0,0], exponent3_data[0,-1,0], colors='k', linestyles='--',zorder=5)
plt.hlines(beta[2], exponent3_data[0,0,0], exponent3_data[0,-1,0], colors='k', linestyles='--',zorder=5)
plt.hlines(beta[4], exponent3_data[0,0,0], exponent3_data[0,-1,0], colors='k', linestyles='--',zorder=5)
plt.hlines(beta[9], exponent3_data[0,0,0], exponent3_data[0,-1,0], colors='k', linestyles='--',zorder=5)


my_y_ticks = [-0.1,-0.05,0.0]
plt.yticks(my_y_ticks)

plt.ylabel(r'$\beta$',fontsize=15,labelpad=-4)

plt.tight_layout()
plt.savefig('plot/mod_u_z4_gap_exponent.pdf',format='pdf',dpi=300)

plt.show() 



 
