import numpy as np
from numpy import pi,exp
from scipy.integrate import solve_ivp
from numba import njit,int32,float64
from functools import partial
import multiprocessing


def envelope_f(n,k1,k2):
    "envelope function"
    f = np.zeros(int(n/2),dtype=np.float64)
    for ii in range(int(n/2)):
        f[ii]=(1.0/(1.0+exp(-k1*(-(ii-(n/2.0-k2))))) )
    return f

# matrix elements of the vertex
@njit(float64[:](int32,int32,float64,float64,float64)) 
def u_m(n,z,phi,u,du):
    u_m  = np.zeros(n-1,np.float64) 
    f_u  = np.zeros(z,np.float64) 
    for j in range(z):
        f_u[j]=-np.sin(2.0*pi*j/z+phi) 
          
    for m in range(0,n,z): 
        for j in range(z):
            u_m[m+j]=u+du*f_u[j]

    return u_m

# matrix elements of the bare hopping matrix
@njit(float64[:](int32,int32,float64,float64,float64)) 
def t_m(n,z,phi,t,dt):
    t_m = np.zeros(n-1,np.float64) 
    f_t  = np.zeros(n,np.float64) 
    for j in range(z):
        f_t[j]=np.cos(2.0*pi*j/z+phi)    
    for m in range(0,n,z): 
        for j in range(z):
            t_m[m+j]=t+dt*f_t[j]
    return t_m

# matrix elements of the onsite potential
@njit(float64[:](int32,int32,float64,float64)) 
def v_m(n,z,phi,v):
    v_m  = np.zeros(n,np.float64) 
    f_v  = np.zeros(z,np.float64) 
    for j in range(z):
        f_v[j]=np.cos(2.0*pi*j/z+phi)
    for m in range(0,n,z): 
        for j in range(z):
            v_m[m+j]=v*f_v[j]

    return v_m



# LDU algorithm for diagonalize tridiagonal matrices
@njit()
def tri(l,y,v_m,t_m,mu,n):  
    a  = np.zeros(n,dtype=np.complex128)
    b  = np.zeros(n-1,dtype=np.complex128)                
    for i in range(n-1):
        a[i] = 1.0j*l + mu - v_m[i] - y[i]
        b[i] = t_m[i]  - y[i+n]
    a[n-1] = 1.0j*l + mu - v_m[n-1]- y[n-1]
    return (a,b)


@njit()
def inv_tri(n,a,b):
    g_dai = np.zeros(n,dtype=np.complex128) 
    g_off = np.zeros(n-1,dtype=np.complex128) 
    up    = np.zeros(n,dtype=np.complex128) 
    un    = np.zeros(n,dtype=np.complex128) 
    dp    = np.zeros(n,dtype=np.complex128) 
    dn    = np.zeros(n,dtype=np.complex128) 
       
    dp[0] = a[0]
    for i in range(n-1):
        up[i]=b[i]/dp[i]
        if (i<(n-1)):
            dp[i+1]=a[i+1]-b[i]*up[i]       
    dn[n-1] = a[n-1]
    for k in range(n-1):
        i=(n-1)-k-1
        un[i]=b[i]/dn[i+1]
        dn[i]=a[i]-b[i]*un[i]       
    g_dai[0]=1.0/dn[i]
    for i in range(n-1):
        g_off[i]   = -un[i]*g_dai[i]
        g_dai[i+1] = g_dai[i]*dp[i]/dn[i+1]
    
    return (g_off,g_dai)


## Density of interacting AAH model from the flow of density respond vertex  

# N: total system size
# Z: size of the sub-lattice
# dt,V: hopping paramter and onsite-potential 
# U,dU: interaction and modulation of the interaction
# phi_v,phi_t,phi_U: phase of the modulation of the V,t and U
# site: targeted position

#L,Z,dt,U,V,dU,mu,phi_v=0.0,phi_t=0.0,phi_U=0.0,t=1.0

# LDU algorithm for diagonalize ABA matrix (called GRG_symm)
@njit()
def grg_symm(n,g_off,g_dia,r_off,r_dia):
    m_dia = np.zeros(n  ,np.complex128) 
    m_off = np.zeros(n-1,np.complex128) 
    lp    = np.zeros(n-1,np.complex128)
    ln    = np.zeros(n-1,np.complex128)
    up    = np.zeros(n-1,np.complex128)
    un    = np.zeros(n-1,np.complex128)
    a     = np.zeros(n  ,np.complex128)
    b     = np.zeros(n-1,np.complex128)
    c     = np.zeros(n-1,np.complex128)
    for i in range(n-1):
        lp[i] = -g_off[i]/g_dia[i+1]
        ln[i] = -g_off[i]/g_dia[i]
        up[i] = -g_off[i]/g_dia[i+1]
        un[i] = -g_off[i]/g_dia[i]            
    for i in range(n):
        a[i] = g_dia[i]*r_dia[i]*g_dia[i]
        if (i<(n-1)): 
            b[i] = g_dia[i]   *r_off[i] *g_dia[i+1]
            c[i] = g_dia[i+1] *r_off[i] *g_dia[i]
    qp = np.zeros(n,np.complex128)
    qn = np.zeros(n,np.complex128)    
    qp[0]   = a[0]
    qn[n-1] = a[n-1]   
    for i in range(n-1):
        qp[i+1] = ln[i]*qp[i]*un[i]      - ln[i]*b[i]   - c[i]*un[i]   + a[i+1] 
        ii=(n-2)-i
        qn[ii]  = up[ii]*qn[ii+1]*lp[ii] - b[ii]*lp[ii] - up[ii]*c[ii] + a[ii] 
    for i in range(n):       
        m_dia[i] = qp[i]- a[i] + qn[i]
        if (i<(n-1)):
            m_off[i] = b[i]-qp[i]*un[i]-up[i]*qn[i+1]+up[i]*c[i]*un[i]
    return (m_dia,m_off)
     
@njit()#(complex128[:](complex128[:],int32))  
def r_dia(y,n):
    rj_dia = np.zeros(n,dtype=np.complex128) 
    for k in range(n):
        rj_dia[k]=y[k+2*n]
    return rj_dia

@njit()#(complex128[:](complex128[:],int32)) 
def r_off(y,n):
    rj_off = np.zeros(n-1,dtype=np.complex128) 
    for k in range(n-1):
        rj_off[k]=y[k+3*n]
    return rj_off


@njit() 
def u_position(n,u,l,k,NN):
    u_position = np.zeros(n,np.complex128) 
    for i in range(n):        
        u_position[i]=u
    return u_position   

@njit() 
def rg_eq_densityvertex(g_off,g_dia,n,z,phi_u,u,du,l,y):
    u_i = u_m(n,z,phi_u,u,du)

    diffeq = np.zeros(4*n+1,dtype=np.complex128)
    ## Diff eq. for the self-energy
    diffeq[0]    =(-u_i[0]/pi)  *g_dia[1].real
    diffeq[n-1]  =(-u_i[n-2]/pi)*g_dia[n-2].real  
    diffeq[n+0]  =( u_i[0]/pi)  *g_off[0].real   
    #Diffeq[2*N-1]=0.0   
    for i in range(1,n-1):

        diffeq[i]  =(-1.0/pi)*(g_dia[i+1]*u_i[i]+g_dia[i-1]*u_i[i-1]).real
        diffeq[i+n]=( u_i[i]/pi)*g_off[i].real
    ## Diff eq. for the density respond vertex
    grjg=grg_symm(n,g_off,g_dia,r_off(y,n),r_dia(y,n))
    ##GRjG[0] diagonal part 
    ##GRjG[1] off-diagonal part 
    diffeq[2*n]   = (-u_i[0]/pi  )*( grjg[0][1]  ).real
    diffeq[3*n-1] = (-u_i[n-2]/pi)*( grjg[0][n-2]).real
    diffeq[3*n]   = ( u_i[0]/pi  )*( grjg[1][0]  ).real
    diffeq[4*n-1] = 0.0
    for k in range(1,n-1):
        diffeq[k+ 2*n] = (-1/pi)*( u_i[k-1]*grjg[0][k-1]+u_i[k]*grjg[0][k+1]  ).real 
        diffeq[k+ 3*n] = ( u_i[k]/pi)*( grjg[1][k]  ).real
    ## Diff eq. for the local density
    diffeq[4*n]=(-1.0/pi)*( ( (np.dot(g_dia,r_dia(y,n))+2.0*np.dot(g_off,r_off(y,n)))).real)
    return diffeq 

@njit() 
def init_vertex(n,u,site):
    init=np.zeros(4*n+1,np.complex128) 
    init[4*n]     =1.0/2.0
    init[2*n+site]=1.0
    return init
    


def vertex_solver(site,n,z,dt,u,v,du,mu,phi_v=0.0,phi_t=0.0,phi_u=0.0,t=1.0):

    @njit()
    def diffeq_vertex(l, y):
        ## y are solutions of differential equations
        ## l is the flow parameter of differential equations
        ## single scale propagator for positive cutoff
        v_i = v_m(n,z,phi_v,v)
        t_i = t_m(n,z,phi_t,t,dt)
        
        
        g_dia = np.zeros(n,np.complex128) 
        g_off = np.zeros(n-1,np.complex128)  
        tridiagonal=tri(l,y,v_i,t_i,mu,n)
        a=tridiagonal[0]
        b=tridiagonal[1]
                   
        inverse_tri=inv_tri(n,a,b)  
        g_off=inverse_tri[0]
        g_dia=inverse_tri[1] 
        

        diffeq = rg_eq_densityvertex(g_off,g_dia,n,z,phi_u,u,du,l,y)
                 
        return diffeq 

    init=init_vertex(n,u,site)
    
    flow=[1e10,1e-8]

    acc=1e-13
    sol=solve_ivp(diffeq_vertex,flow,init,method='RK45',rtol=acc,atol=1e-13)
    return sol



def rho(n,z,dt,u,v,du,mu,phi_v,phi_t,phi_u,site):
    sol=vertex_solver(site,n,z,dt,u,v,du,mu,phi_v,phi_t,phi_u)
    return sol    



def rho_vertex_parallel(range_boundary,n,z,dt,u,v,du,mu,phi_v,phi_t,phi_u):
    lattic    = np.arange(range_boundary)
    rho_vertex= np.zeros(range_boundary,dtype=np.float64)
    
    cores = multiprocessing.cpu_count()
    pool  = multiprocessing.Pool(processes=cores)
    func  = partial(rho,n,z,dt,u,v,du,mu,phi_v,phi_t,phi_u)
    result= pool.map(func, lattic)
    
    for i in range(range_boundary):
        rho_vertex[i]=(result[i].y[4*n,-1]).real
    return rho_vertex


