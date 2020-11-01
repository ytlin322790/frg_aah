import numpy as np
from tool_aah_solver import vertex_solver
import sys
from parameters_bc import *

i=int(sys.argv[1])
ii=int(sys.argv[2])
site=int(sys.argv[3])


MU=0.0
U_0=U_LIST[i]

V=V_LIST[ii]
DT=DT_LIST[ii]



density_vertex=np.zeros(N_SUB,dtype=np.float64)

for m in range(N_SUB):
    rho_vertex       =vertex_solver(site+m,N,Z,DT,U_0,V,DU,MU,phi_v=PHI_V,phi_t=PHI_T,phi_u=0.0,t=1.0)
    density_vertex[m]=rho_vertex.y[4*N,-1].real


np.save('data/LD(U=%g,ii=%g,site=%g).npy'%(U_0,ii,site),density_vertex)



