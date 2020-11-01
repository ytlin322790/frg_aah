from numpy import pi
#%%
# system size
L=4000#1000
N=L
TARGET=3000.0
# number of sublattice
Z=4

# onsite potential
V=0.015
# hopping parameters
DT=0.02



# phase of modulation
PHI_V=-0.45-pi/4.0#*np.pi
PHI_T=-0.45-pi/4.0#*np.pi
PHI_U=0.0



DU=0.0#1
U=0.08#0.075#125


# scale of the mu

M =0.001



NN=5
NNN=41

#P1=1.0
#P2=5.0
#P3=10.0
#P4=50.0
#P5=100.0

P1=200.0
P2=250.0
P3=300.0
P4=350.0
P5=1000.0

MU_UPPER_1=10.0
MU_LOWER_1=1.0

MU_UPPER_2=10.0
MU_LOWER_2=1.0

MU_UPPER_3=10.0
MU_LOWER_3=1.0

MU_UPPER_4=10.0
MU_LOWER_4=1.0

MU_UPPER_5=25.0
MU_LOWER_5=1.0


FILLING=0.75
