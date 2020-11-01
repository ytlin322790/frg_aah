import numpy as np
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import sys 
#sys.path.append('/Users/yentinLin/Desktop/frg_aah_model')
sys.path.append('..')
#from parameters_aah_z4_3sqrt_filling import *
from tool.frg_aah_greenfunction import FrgGreenData
#from frg_aah_tool import eff_gap_z4,search_mu_green,generate_boundarycharge,dQdgamma
from tool.frg_aah_parallel import find_mu_aah_z4_parallel,find_mu_mod_u_z4_parallel

from tool.tool_general import loglogder


#sys.path.append('/Users/yentinLin/Desktop/frg_aah_model/project2_interacting_aah_model_z=4/parameter_data')
sys.path.append('/frg_aah_model/project2_interacting_aah_model_z=4')

from parameter_data.parameters_bc_3qrt_filling import *



#%%

p1=125
p2=150
p3=175
p4=200
p5=250


a=find_mu_mod_u_z4_parallel(n_phi=NNN,n=N,z=Z,dt=DT,u=U,v=V,du=DU*p1,pn=TARGET,mu_upper=3.0,mu_lower=-1e-6)
#print(mu_list_a)

b=find_mu_mod_u_z4_parallel(n_phi=NNN,n=N,z=Z,dt=DT,u=U,v=V,du=DU*p2,pn=TARGET,mu_upper=3.0,mu_lower=-1e-6)
#print(mu_list_b)

c=find_mu_mod_u_z4_parallel(n_phi=NNN,n=N,z=Z,dt=DT,u=U,v=V,du=DU*p3,pn=TARGET,mu_upper=4.0,mu_lower=-1e-6)
#print(mu_list_c)

d=find_mu_mod_u_z4_parallel(n_phi=NNN,n=N,z=Z,dt=DT,u=U,v=V,du=DU*p4,pn=TARGET,mu_upper=4.0,mu_lower=-1e-6)
#print(mu_list_d)

e=find_mu_mod_u_z4_parallel(n_phi=NNN,n=N,z=Z,dt=DT,u=U,v=V,du=DU*p5,pn=TARGET,mu_upper=5.0,mu_lower=-1e-6)
#print(mu_list_e)

mu_list_a=a
mu_list_b=b
mu_list_c=c
mu_list_d=d
mu_list_e=e

#%%

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

#%%

r'''
np.save('frg_data/mu_qrt_v=%g_dt=%g.npy'%(V,DT),mu_list_a)
np.save('frg_data/bc_qrt_v=%g_dt=%g.npy'%(V,DT),bc_list_a)

np.save('frg_data/mu_qrt_v=%g_dt=%g.npy'%(5.0*V,5.0*DT),mu_list_b)
np.save('frg_data/bc_qrt_v=%g_dt=%g.npy'%(5.0*V,5.0*DT),bc_list_b)


np.save('frg_data/mu_qrt_v=%g_dt=%g.npy'%(10.0*V,10.0*DT),mu_list_c)
np.save('frg_data/bc_qrt_v=%g_dt=%g.npy'%(10.0*V,10.0*DT),bc_list_c)


np.save('frg_data/mu_qrt_v=%g_dt=%g.npy'%(50.0*V,50.0*DT),mu_list_d)
np.save('frg_data/bc_qrt_v=%g_dt=%g.npy'%(50.0*V,50.0*DT),bc_list_d)


np.save('frg_data/mu_qrt_v=%g_dt=%g.npy'%(100.0*V,100.0*DT),mu_list_e)
np.save('frg_data/bc_qrt_v=%g_dt=%g.npy'%(100.0*V,100.0*DT),bc_list_e)
'''

r'''
np.save('frg_data/mu_qrt_dU=%g.npy'%(DU*p),mu_list_a)
np.save('frg_data/bc_qrt_dU=%g.npy'%(DU*p),bc_list_a)

np.save('frg_data/mu_qrt_dU=%g.npy'%(DU*pp),mu_list_b)
np.save('frg_data/bc_qrt_dU=%g.npy'%(DU*pp),bc_list_b)

np.save('frg_data/mu_qrt_dU=%g.npy'%(DU*ppp),mu_list_c)
np.save('frg_data/bc_qrt_dU=%g.npy'%(DU*ppp),bc_list_c)

np.save('frg_data/mu_qrt_dU=%g.npy'%(DU*pppp),mu_list_d)
np.save('frg_data/bc_qrt_dU=%g.npy'%(DU*pppp),bc_list_d)

np.save('frg_data/mu_qrt_dU=%g.npy'%(DU*ppppp),mu_list_e)
np.save('frg_data/bc_qrt_dU=%g.npy'%(DU*ppppp),bc_list_e)

'''
#r'''

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

#'''

#%%

phi_list=np.zeros(NNN,dtype=np.float64)

for i in range(NNN):
    phi=i/NNN+0.001/2.0*np.pi
    phi_list[i]=phi
    
r'''    
np.load('frg_data/mu_3qrt_v=%g_dt=%g.npy'%(V,DT))
A=np.load('frg_data/bc_3qrt_v=%g_dt=%g.npy'%(V,DT))

np.load('frg_data/mu_3qrt_v=%g_dt=%g.npy'%(5.0*V,5.0*DT))
B=np.load('frg_data/bc_3qrt_v=%g_dt=%g.npy'%(5.0*V,5.0*DT))


np.load('frg_data/mu_3qrt_v=%g_dt=%g.npy'%(10.0*V,10.0*DT))
C=np.load('frg_data/bc_3qrt_v=%g_dt=%g.npy'%(10.0*V,10.0*DT))


np.load('frg_data/mu_3qrt_v=%g_dt=%g.npy'%(50.0*V,50.0*DT))
D=np.load('frg_data/bc_3qrt_v=%g_dt=%g.npy'%(50.0*V,50.0*DT))


np.load('frg_data/mu_3qrt_v=%g_dt=%g.npy'%(100.0*V,100.0*DT))
E=np.load('frg_data/bc_3qrt_v=%g_dt=%g.npy'%(100.0*V,100.0*DT)) 
'''

r'''
np.load('frg_data/mu_qrt_dU=%g.npy'%(DU*p))
A=np.load('frg_data/bc_qrt_dU=%g.npy'%(DU*p))

np.load('frg_data/mu_qrt_dU=%g.npy'%(DU*pp))
B=np.load('frg_data/bc_qrt_dU=%g.npy'%(DU*pp))

np.load('frg_data/mu_qrt_dU=%g.npy'%(DU*ppp))
C=np.load('frg_data/bc_qrt_dU=%g.npy'%(DU*ppp),)

np.load('frg_data/mu_qrt_dU=%g.npy'%(DU*pppp))
D=np.load('frg_data/bc_qrt_dU=%g.npy'%(DU*pppp))

np.load('frg_data/mu_qrt_dU=%g.npy'%(DU*ppppp))
E=np.load('frg_data/bc_qrt_dU=%g.npy'%(DU*ppppp))
'''
#r'''
np.load('frg_data/mu_3qrt_dU=%g.npy'%(DU*p))
A=np.load('frg_data/bc_3qrt_dU=%g.npy'%(DU*p))

np.load('frg_data/mu_3qrt_dU=%g.npy'%(DU*pp))
B=np.load('frg_data/bc_3qrt_dU=%g.npy'%(DU*pp))

np.load('frg_data/mu_3qrt_dU=%g.npy'%(DU*ppp))
C=np.load('frg_data/bc_3qrt_dU=%g.npy'%(DU*ppp),)

np.load('frg_data/mu_3qrt_dU=%g.npy'%(DU*pppp))
D=np.load('frg_data/bc_3qrt_dU=%g.npy'%(DU*pppp))

np.load('frg_data/mu_3qrt_dU=%g.npy'%(DU*ppppp))
E=np.load('frg_data/bc_3qrt_dU=%g.npy'%(DU*ppppp))
#'''
   
#%%
fig,ax=plt.subplots()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.plot(phi_list,A,'.',label=r"V=%g dt=%g"%(V,DT),rasterized=True)#
#plt.plot(phi_list,B,'.',label=r"V=%g dt=%g"%(5.0*V,5.0*DT),rasterized=True)#
#plt.plot(phi_list,C,'.',label=r"V=%g dt=%g"%(10.0*V,10.0*DT),rasterized=True)#
#plt.plot(phi_list,D,'.',label=r"V=%g dt=%g"%(50.0*V,50.0*DT),rasterized=True)#
#plt.plot(phi_list,E,'.',label=r"V=%g dt=%g"%(100.0*V,100.0*DT),rasterized=True)#
plt.plot(phi_list,A,'.',label=r"dU=%g"%(p*DU),rasterized=True)#
plt.plot(phi_list,B,'.',label=r"dU=%g"%(pp*DU),rasterized=True)#
plt.plot(phi_list,C,'.',label=r"dU=%g"%(ppp*DU),rasterized=True)#
plt.plot(phi_list,D,'.',label=r"dU=%g"%(pppp*DU),rasterized=True)#
plt.plot(phi_list,E,'.',label=r"dU=%g"%(ppppp*DU),rasterized=True)#
#plt.plot(np.arange(0,len(sol_FRG.y[:,-1]),1),sol_FRG.y[:,-1],'o-',label="old clean")#
plt.legend(loc='best')
#ax.set_yscale('log')
plt.ylim([-0.55,0.55])
my_x_ticks = [0.0,0.25,0.5,0.75,1.0]
plt.xticks(my_x_ticks)
my_y_ticks = [-0.5,-0.25,0.0,0.25,0.5]
plt.yticks(my_y_ticks)
plt.xlabel(r'$\gamma/2\pi $',fontsize=18)
plt.ylabel(r'$Q_B$',fontsize=18)
#plt.ylabel(r'$t^{FRG}_{1,FO}e^{2j/ξ_{test}}$')
plt.tight_layout()
#plt.savefig('bc_mod_u_3sqrtfilling.pdf',format='pdf',dpi=300)
plt.show() 