<p align="center">
  <a href="#about">About</a> ◈
  <a href="#prerequisites">Prerequisites</a> ◈
  <a href="#Getting-started">Getting-started</a> ◈
  <a href="#Local-Density-with-Green-function-approach">Local Density with Green function approach</a> ◈
  <a href="#Local-Density-with-Vertex-flow-approach">Local Density with Vertex flow approach</a> 
</p>

---

## About

This is the functional renormalization group (fRG) numerical solver on 1d spinless fermionic lattice model, which allows you to do the following:

* study the correlated fermionic problem on 1d spinless fermionic lattice model in the weak coupling limit.
* generate the real space and k-space data for one-particle irreducible (1PI) vertex, the self-energy. 
* generate the real spcae and k-space data for local density in large system size.
* generate the local spectrum function.

The models which can be studied by this framework are given as:

* spinless fermion model 
* interacting Su-Schrieffer-Heeger (SSH) model and Rice-Mele (RM) model. 
* generalized Aubry-Andr\'e-Harper (AAH) model with Z periodicity. 
* modulated interaction model with Z periodicity. 

The essential data is fRG self-energy (one-particle irreducible vertex).
The data for the local denisty is calculated in two method:
  1. Effecitve single particle method: The RG flow of 1PI vertex. fRG self-energy plus single particle green's function, and
  2. Many body method: The RG flow of 1PI vertex and the total particle density. 

The data for the local spectrum function is obtained by the excat numerical diagonalization of effective single particle hamiltonain with fRG self-energy.

The algorithm:

We use the Runge-Kutta solvers of orders 4 and 5 (RK45) to implement the renormalization group equations which is a set of differential equations. 
Moreoever, we use scipy.integrate sub-package to solve ordinary differential equation integrator.
Due to the for loop structure in the solver, we also add the Numba JIT compiler to speed up the process of implementing the for loop.

To implement this fRG code, one need to invert the tridigonal matrix in the right hand side of the RG equations. 
The computation effort of inverting the generic matrix by standard methods is O(L^2).
However, for the nearest neighbor hoppings and the interaction, only the tridigonal part of the single scale propagator is required.
Therefore, by the "LDU factorization", the computation effort can be reduced in O(L). This is the advantage of the functional RG method over another numerical method, such as DMRG. 
More detail can be found in Ref.['https://journals.aps.org/prb/abstract/10.1103/PhysRevB.70.075102'].

## Prerequisites

To implement this functional RG code, you need Python version 3.8 or 3.7, numpy, scipy and numba installed on your device.
For the density with functional RG density respond vertex flow in large system size, one need to implement the numerics in the cluster.

To download numpy and scipy, you can do the following:

```bash
pip install numpy
pip install scipy
```

More detail of the numpy and scipy can be found in the website 'https://numpy.org' and 'https://www.scipy.org'.

To download numba,  do the following:
```bash
pip install numba
```

More detail about the numba, one can visit the website 'https://numba.pydata.org'.
## Getting-started

The essential object in the fRG formalism is the FRG one-particle irreducible (1PI) vertex.
We use the generalized Aubry-Andr\'e-Harper (AAH) model with Z periodicity as a targeted model for the following explanation.  
One can use the data of 1PI vertex to calculate the observable such as single particle spectral function the local density and so on.

The non-interacting part of the AAH model, the Hamiltonian contains the modulation of on-site potentail and the hopping parameters. 

The total system sites is given by N,  
the number of the sublattices is Z,  
the hopping parameter is t=1 by default,  
the modulation of the hopping and the onsite potential is DT and V,  
the phase of the hopping and the onsite potential is PHI_T and PHI_V, PHI_T=PHI_V=0.0 by default,  
rhe chemical potential is given by MU, and  
the strength of the interaction is given by U.  

In order to obtain the real space functional RG data, one should 

```py
import numpy as np
from tool.green_function import FrgGreenData

# Create a new instance of the functional RG self-energy.
# Take AAH model for example. 

N=100
Z=4
DT=V=0.05
MU=0.03


p=FrgGreenData()
p.frg_obc(n=N,z=Z,dt=DT,u=U,v=V,du=0.0,mu=MU)
frg_self_energy=p.get_selfenergy()
print(frg_self_energy)
```


## Local Density with Green function approach

After implementing the part of the fRG self-energy, one can obtain the local density (total particle density) in all system by integrating the green function over Mutsubara frequency.

```py
    
p=FrgGreenData()
p.frg_obc(n=N,z=Z,dt=DT,u=U,v=V,du=0.0,mu=MU)
p.local_density(n=N,z=Z,dt=DT,u=U,v=V,du=DU,mu=MU)
rho=p.get_local_density()

print(rho)
```


## Local Density with Vertex flow approach

In order to ontain the more accurate and self-consistent result in functional RG formalism, one need to include the flow of density respond vertex. Note that this method is different from the Green function approach, one only obtain result in single position j in the end of the flow, for example, 
```py
import numpy as np
from tool.vertex import FrgVertexData

# Create a new instance of the functional RG self-energy + vertex respond vertex
# Take AAH model for example. 

N=100
Z=4
DT=V=0.05
MU=0.0
j=0

p=FrgVertexData()
p.vertex_solver(site=j,n=N,z=Z,dt=DT,u=U,v=V,du=DU,mu=MU)
rho_vertex=p.get_frg_vertexflow_data()
print(rho_vertex)
```




