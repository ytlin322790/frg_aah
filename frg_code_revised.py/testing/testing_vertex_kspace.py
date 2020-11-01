import sys 
import numpy as np
sys.path.append('..')
from tool.frg_aah_kspace import frg_vertex_z4_bulk_solver
from tool.frg_aah_vertex import FrgVertexData

Z=4
DT=0.01
V=0.13
MU=0.0
U=0.125
DU=0.0


N=50000

z=Z
dt=DT
v=V
mu=MU
u=U
du=DU
n=N
site_a=0
site_b=1
site_c=2
site_d=3

rho_sol_a=frg_vertex_z4_bulk_solver(site_a,z,dt,v,mu,u,du)
rho_sol_b=frg_vertex_z4_bulk_solver(site_b,z,dt,v,mu,u,du)
rho_sol_c=frg_vertex_z4_bulk_solver(site_c,z,dt,v,mu,u,du)
rho_sol_d=frg_vertex_z4_bulk_solver(site_d,z,dt,v,mu,u,du)

print('-----')
print(rho_sol_a.y[-1,-1])
print(rho_sol_b.y[-1,-1])
print(rho_sol_c.y[-1,-1])
print(rho_sol_d.y[-1,-1])
print('-----')


p=FrgVertexData()
p.vertex_solver(site=site_a+20000,n=N,z=Z,dt=DT,u=U,v=V,du=DU,mu=MU)

pp=FrgVertexData()
pp.vertex_solver(site=site_b+20000,n=N,z=Z,dt=DT,u=U,v=V,du=DU,mu=MU)

ppp=FrgVertexData()
ppp.vertex_solver(site=site_c+20000,n=N,z=Z,dt=DT,u=U,v=V,du=DU,mu=MU)

pppp=FrgVertexData()
pppp.vertex_solver(site=site_d+20000,n=N,z=Z,dt=DT,u=U,v=V,du=DU,mu=MU)

ans_1=p.get_local_density()
ans_2=pp.get_local_density()
ans_3=ppp.get_local_density()
ans_4=pppp.get_local_density()
print(ans_1.real)
print(ans_2.real)
print(ans_3.real)
print(ans_4.real)

