import numpy as np
from numpy import sin,cos,arccos
import sys 
import matplotlib.pyplot as plt
sys.path.append('../..')
from tool.tool_general import loglogder
sys.path.append('project1/effective_gap')
from parameter.parameters_gap_3qrt_filling_trial import *

free_mu_data=np.load('data/mu_free_gap_3qrt_v=%g_dt=%g.npy'%(V_0,DT_0))
frg_bulk_gap=np.load('data/frg_gap_3qrt_v=%g_dt=%g.npy'%(V_0,DT_0))
free_bulk_gap=np.load('data/free_gap_3qrt_v=%g_dt=%g.npy'%(V_0,DT_0))

u_ref=np.zeros(N_U,dtype=np.float64)
u_data=np.zeros(N_U,dtype=np.float64)
gap1_data=np.zeros(N_U,dtype=np.float64)
gap2_data=np.zeros(N_U,dtype=np.float64)
gap3_data=np.zeros(N_U,dtype=np.float64)
gap_pbc_data=np.zeros(N_U,dtype=np.float64)

for i in range(N_U):
    U=U_0*i
    u_data[i]=U
    
    mu_0=free_mu_data[i][0]
    kF=arccos(-mu_0/2.0)

    u_ref[i] = -U*(1.0-cos(2.0*kF))/(2.0*np.pi*sin(kF))

    AA=loglogder(free_bulk_gap[i,:,0],np.abs(frg_bulk_gap[i,:,0]/free_bulk_gap[i,:,0]))
    BB=loglogder(free_bulk_gap[i,:,1],np.abs(frg_bulk_gap[i,:,1]/free_bulk_gap[i,:,1]))
    CC=loglogder(free_bulk_gap[i,:,2],np.abs(frg_bulk_gap[i,:,2]/free_bulk_gap[i,:,2]))
    

    gap1_data[i]=np.average(AA[1])
    gap2_data[i]=np.average(BB[1])
    gap3_data[i]=np.average(CC[1])

    print(u_ref[i])
    print("gap_1:",gap1_data[i])
    print("gap_2:",gap2_data[i])
    print("gap_3:",gap3_data[i])
    


exponent_gap=[u_data,gap3_data]
exponent_u=[u_data,u_ref]
np.save('data/exponent_gap_3qrt_v=%g_dt=%g.npy'%(V_0,DT_0),exponent_gap)
np.save('data/exponent_fieldtheory_3qrt_v=%g_dt=%g.npy'%(V_0,DT_0),exponent_u)

