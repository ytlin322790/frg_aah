import numpy as np
from numpy import cos,exp
from parameters_bc import *
from tool_aah_solver import envelope_f


density=np.zeros([NNN,N],dtype=np.float64)



density=np.zeros([NN,NNN,np.int(N/2)],dtype=np.float64)
qb=np.zeros([NN,NNN],dtype=np.float64)
f=envelope_f(N,k1=0.15,k2=100)
for i in range(NN):
    U_0=U_LIST[i]
    for ii in range(N_PHI):
        V=V_LIST[ii]
        DT=DT_LIST[ii]      
        for site in range(0,np.int(N/2),N_SUB):
            density_part=np.load('data/LD(U=%g,ii=%g,site=%g).npy'%(U_0,ii,site))
            for m in range(N_SUB):
                density[i,ii,site+m]=density_part[m]

    
        FILLING_VERTEX=np.average(density[i,ii,np.int(N/2)-200:np.int(N/2):1])
        print(FILLING_VERTEX)
        print('total number (vertex):',np.sum(density[i,ii,:]))
        bc=np.sum((density[i,ii,:]-FILLING_VERTEX)*f)
        qb[i,ii]=bc
        print(bc)
        print('============')

np.save('QB_aah_halffilling_u.npy',qb)


