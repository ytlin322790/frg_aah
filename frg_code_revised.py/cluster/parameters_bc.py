from numpy import pi

# system size
L=4000
N=L
TARGET=2000.0
# number of sublattice
Z=4
# onsite potential
V=0.15
# hopping parameters
DT=0.0#2
# phase of modulation
PHI_V=5.0*pi/6.0
PHI_T=5.0*pi/6.0
PHI_U=0.0

DU=0.0

U_LIST=[0.000,0.008,0.0128,0.020480,0.032768,0.0524288,0.08388608,0.134217728,0.2147483648,0.34359738368,0.549755813888]
MU=0.0
NN=11
NNN=N_PHI=6
N_SUB=400
V_LIST=[0.5, 0.75, 0.5, 0.75, 0.5, 0.75]
DT_LIST=[0.1, 0.15,0.25, 0.375, 0.4, 0.6]
FILLING=0.5
