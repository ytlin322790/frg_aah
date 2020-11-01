#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numba import njit
import numpy as np
from numpy import linalg as LA
from numpy import pi,exp
from scipy.integrate import solve_ivp
from tool.tool_aah_solver import u_m,t_m,v_m,tri,inv_tri


class FrgGreenData():
    """ The functional renormalization group (fRG) on 1d spinless fermionic lattice model.
    The essntial data is fRG self-energy.
    The data for the local denisty is calculated from fRG self-energy plus single particle green's function.
    """

    def __init__(self):
        """ Create a new fRG data.

        _frg_flow_data     the name of the all of fRG self-energy during the RG-flow 
        _frg_selfenergy    the name of the fRG self-energy in the end of the flow
        _local_density     the name of the local density for given fRG self-energy
        _boundary_charge   the name of the boundary charge for given local density 
        _bulk_parameters   the name of the bulk renormalized parameters
        """
        self._frg_flow_data=0.0
        self._frg_selfenergy=0.0
        self._local_density=0.0
        self._boundary_charge=0.0
        self._bulk_parameters=0.0
        self._bulk_band=0.0
        self._gap=0.0

    def get_selfenergy(self):
        """Return the fRG self-energy in the end of the flow."""
        return self._frg_selfenergy

    def get_flow_data(self):
        """Retrun the all of the data concerning frg self-energy."""
        return self._frg_flow_data

    def get_local_density(self):
        """Retrun the local density obtained from green function method."""
        return self._local_density

    def get_boundary_charge(self):
        """Retrun the boundary charge obtained green function method."""
        return self._boundary_charge

    def get_bulk_parameters(self):
        """Retrun the bulk_parameters renormalized by frg self-energy."""
        return self._bulk_parameters

    def get_bulk_band(self):
        """Retrun the effecitve bulk spectrum."""
        return self._bulk_band

    def get_gap(self):
        """Retrun the effecitve bulk spectrum."""
        return self._gap

    def frg_obc(self,n,z,dt,u,v,du,mu,
                phi_v=0.0,phi_t=0.0,phi_u=0.0,t=1.0):
        """ The solver generating fRG self-energy.

        Input Parameters:
        ----------
        n : integer
            The total system size.
        z : integer
            The number of sub-lattice.
        dt : float
            The magnitude of the hopping modulation.
        u : float
            The magnitude of the interaction.
        v : float
            The magnitude of onsite potential.
        du : float 
            The magnitude of modulation of the interaction.
        mu : float
            The chemical potential.
        phi_v, phi_t ,phi_u : float
            The phase of the onsite-potential, hopping and the interaction.
        t : float
            4t is the band-width.

        Output:
        ----------
        The fRG self-energy during the RG flow.
        """
        v_i = v_m(n,z,phi_v,v)
        t_i = t_m(n,z,phi_t,t,dt)
        u_i = u_m(n,z,phi_u,u,du)

        @njit()
        def rg_eq(g_off,g_dia,n):
            """Generator of right hand side of the RG equations:

            Input:
            ------------
            g_off, g_dia: float
            (offdigonal and digonal part of the green function) 
            
            n: integer
            the number of total system size.

            Output:
            ------------
            right hand side of the RG equations
            """

            diffeq=np.zeros(2*n,dtype=np.float64)
            diffeq[0]=(-u_i[0]/pi)*g_dia[1].real
            diffeq[n-1]=(-u_i[n-2]/pi)*g_dia[n-2].real  
            diffeq[n]=(u_i[0]/pi)*g_off[0].real 
        
            for ii in range(1,n-1):
                diffeq[ii]=(-1.0/pi)*(g_dia[ii+1]*u_i[ii]+g_dia[ii-1]*u_i[ii-1]).real
                diffeq[ii+n]=(1.0/pi)*(g_off[ii]*u_i[ii]).real

            return diffeq 
    
        @njit()
        def diffeq(l, y):
            ## y are solutions of differential equations: set
            ## l is the flow parameter of differential equations: float
            ## single scale propagator for positive cutoff
            g_dia=np.zeros(n,np.complex128) 
            g_off=np.zeros(n-1,np.complex128)  
            tridiagonal=tri(l,y,v_i,t_i,mu,n)
            a=tridiagonal[0]
            b=tridiagonal[1]
                   
            inverse_tri=inv_tri(n,a,b)  
            g_off=inverse_tri[0]
            g_dia=inverse_tri[1] 

            diffeq=rg_eq(g_off,g_dia,n)
            return diffeq  

        #initial condition of the solution
        init=np.zeros(2*n,dtype="float") 

        #starting and end points of the flow parameter
        #flow=[1e15,1e-15]
        flow=[1e8,1e-11]

        #relative accuracy in RK45 solver
        acc=2.220446049250313e-14
        
        #sol=solve_ivp(diffeq,flow,init,method='RK45',rtol=acc,atol=1e-20)
        sol=solve_ivp(diffeq,flow,init,method='RK45',rtol=acc,atol=1e-15)
                
        self._frg_flow_data=sol
        self._frg_selfenergy=sol.y[:,-1]

        return sol
    
     
    def local_density(self,n,z,dt,u,v,du,mu,
                      phi_v=0.0,phi_t=0.0,phi_u=0.0,t=1.0):
        """ The solver generating local density using green function method.

        Input Parameters:
        ----------
        n : integer
            The total system size.
        z : integer
            The number of sub-lattice.
        dt : float
            The magnitude of the hopping modulation.
        u : float
            The magnitude of the interaction.
        v : float
            The magnitude of onsite potential.
        du : float 
            The magnitude of modulation of the interaction.
        mu : float
            The chemical potential.
        phi_v, phi_t ,phi_u : float
            The phase of the onsite-potential, hopping and the interaction.
        t : float
            4t is the band-width.

        Output:
        ----------
        The fRG self-energy during the RG flow.
        """    
        v_i=v_m(n,z,phi_v,v)
        t_i=t_m(n,z,phi_t,t,dt)
        #u_i=u_m(n,z,phi_u,u,du)

        @njit()
        def diffeq_density(l,n,t_i,v_i,mu,sol):
            """Generator of right hand side of the diffential equation of local density.
            Input:
            ------------
            
            l : float
                the parameter of differential equations
            n : integer
                the number of total system size.
            t_i : set
                one period of the hopping parameters
            v_i : set 
                one period of the onsite potential
            mu: float
                chemical potential
            sol: set
                FRG self-energy in the end of the flow.

            Output:
            ------------
            right hand side of the RG equations
            """

            g_dia=np.zeros(n,dtype=np.complex128) 
            a=np.zeros(n,dtype=np.complex128)
            b=np.zeros(n-1,dtype=np.complex128)
            for i in range(n-1):
                a[i]=1.0j*l+mu-v_i[i]-sol[i]
                b[i]=t_i[i]-sol[i+n]
            a[n-1]=1.0j*l+mu-v_i[n-1]-sol[n-1]

            inverse_tri=inv_tri(n,a,b)  
            g_dia=inverse_tri[1] 
            diffeq=np.zeros(n,dtype=np.float64)
            for i in range(n):        
                diffeq[i]=(1.0/pi)*g_dia[i].real  
            return diffeq 

        def diff_eq_density(l,y):  
            diffeq=diffeq_density(l,n,t_i,v_i,mu,self._frg_selfenergy)               
            return diffeq   

        #initial condition
        init=np.zeros(n,dtype="float")
        for i in range(n):
            init[i]=0.5     

        #flow parameter
        #flow=[1e-30,1e30] 
        flow=[1e-20,1e20] 

        # relative accuarcy of RK45 solver
        acc =2.220446049250313e-14

        #density = solve_ivp(diff_eq_density,flow,init,method='RK45',rtol=acc,atol=1e-20)  
        density = solve_ivp(diff_eq_density,flow,init,method='RK45',rtol=acc,atol=1e-15)  

        self._local_density=density.y[:,-1]
        
        return density

    def boundary_charge(self,n,filling,k1=0.15,k2=100):
        """ The solver generating boundary charge for given local density.

        Input Parameters:
        ----------
        n : integer
            The total system size.
        filling : float
            The filling of total system: total particle/total system size.
        k1 : float
            Inverse of the decay length of envelope function.
        k2 : float
            Starting position of the decay.


        Output:
        ----------
        boundary charge: float
        """    
        def envelope_f(n,k1,k2):
            "envelope function"
            f = np.zeros(int(n/2),dtype="float") 
            for ii in range(int(n/2)):
                f[ii]=(1.0/(1.0+exp(-k1*(-(ii-(n/2.0-k2))))) )
            return f

        f=envelope_f(n,k1=k1,k2=k2)
        local_density=self.get_local_density()
        bc=np.sum((local_density[0:int(n/2):1]-filling)*f) 
        #print(bc)
        self._boundary_charge=bc

        return bc

    def bulk_parameters(self,n,z,dt,u,du,v,mu,phi_v=0.0,phi_t=0.0,phi_u=0.0,t=1.0,ave=10):  
        """ The solver generating bulk average renormalized parameters.

        Input Parameters:
        ----------
        n : integer
            The total system size.
        z : integer
            The number of sub-lattice.
        dt : float
            The magnitude of the hopping modulation.
        u : float
            The magnitude of the interaction.
        v : float
            The magnitude of onsite potential.
        du : float 
            The magnitude of modulation of the interaction.
        mu : float
            The chemical potential.
        phi_v, phi_t ,phi_u : float
            The phase of the onsite-potential, hopping and the interaction.
        t : float
            4t is the band-width.
        ave : integer
            The total number of positions which are taken into bulk averaging. 
        Output:
        ----------
        The average bulk parameters.
        """    

        if (n%2) == 1:
            raise ValueError('number of total lattices needs to even number.')
        else:
            pass
        if (np.int(n/2)%z)==0:
            pass
        else:
            raise ValueError('Half the system size needs to be multiple of Z.')
        if (n<400):
            raise ValueError('system size is too small.')
        elif (400<n<1000):
            print('system size might be too small. Please check the finite size effect.')
        else:
            pass
        
        sol=self._frg_selfenergy

        #print('frg self-energy:',sol[n:n+10:1])
        #print(sol)
        frg_onsite_selfenergy=sol[0:np.int(n/2):1]
        frg_off_selfenergy=sol[n:n+np.int(n/2):1]

        #print('frg self-energy:',frg_off_selfenergy)

        # ren hopping and onsite potential
        ren_t=np.zeros(z,np.float64) 
        ren_v=np.zeros(z,np.float64) 
        self_t=np.zeros(z,np.float64) 
        self_v=np.zeros(z,np.float64)  
    
        for i in range(z):
            a=0.0
            b=0.0
            for ii in range(ave):
                a+=frg_onsite_selfenergy[-z*(ave)+ii*z+i]/ave
                b+=frg_off_selfenergy[-z*(ave)+ii*z+i]/ave
                
            #print('------')
            self_v[i]=a
            self_t[i]=b
            #print('B:',b)
        v_i=v_m(n,z,phi_v,v)
        t_i=t_m(n,z,phi_t,t,dt)
        #print('v_i:',v_i)
        #print('t_i:',t_i)

        for i in range(z):
            ren_t[i]=t_i[i]-self_t[i]
            ren_v[i]=v_i[i]+self_v[i]

        #print('ren_v:',ren_v)
        #print('ren_t:',ren_t)
        self._bulk_parameters=[ren_v,ren_t]
        return [ren_v,ren_t]

    def bulk_band_aah(self,n,z,tj,vj):
        """ The generator of bulk spectrum in k-space.

        Input Parameters:
        ----------
        n : integer
            The total system size.
        z : integer
            The number of sub-lattice.
        tj : set
            renormalized hopping parameters.
        vj : set
            renormalized onsite parameters.
        Output:
        ----------
        The bulk spectrum in k-space
        """    
        def bulk_H(n,tj,vj,k):
            H=np.zeros((z,z),dtype=np.complex128) 
            for i in range(z-1):
                H[i,i]=vj[i] 
                H[i,i+1]=H[i+1,i]=-tj[i]      
            H[z-1,z-1]=vj[z-1] 
            H[0,z-1]=-tj[z-1]*exp(-1.0j*k) 
            H[z-1,0]=-tj[z-1]*exp(1.0j*k)  
            w,v=LA.eigh(H)
            return w.real

        bz=[-pi+2.0*pi*x/n for x in range(n)]
        band=np.zeros([n,z],dtype=np.float64)
        for i in range(n):
            k=bz[i]
            H=bulk_H(z,tj,vj,k)
            for ii in range(z):
                band[i,ii]=H[ii].real

        self._bulk_band=band
        return band

    def bulk_gap_aah_z4(self,n,z,t_j,v_j):
        """ The generator of effecitve single particle gap in effecitve AAH Z=4 model.

        Input Parameters:
        ----------
        n : integer
            The total system size.
        z : integer
            The number of sub-lattice.
        tj : set
            renormalized hopping parameters.
        vj : set
            renormalized onsite parameters.
        Output:
        ----------
        The bulk gap
        """  
        #print(t_j)
        #print(v_j)
        def bulk_h(n,tj,vj,k):
            H=np.zeros((z,z),dtype=np.complex128) 
            for i in range(z-1):
                H[i,i]=vj[i] 
                H[i,i+1]=H[i+1,i]=-tj[i]      
            H[z-1,z-1]=vj[z-1] 
            H[0,z-1]=-tj[z-1]*exp(-1.0j*k) 
            H[z-1,0]=-tj[z-1]*exp(1.0j*k)  
            w,v=LA.eigh(H)
            return w.real
        #print('!!!!')
        gap=np.zeros(3,dtype=np.float64)


        H_k0=bulk_h(z,t_j,v_j,0.0)
        H_kpi=bulk_h(z,t_j,v_j,np.pi)
        #print(H_k0)
        #print(H_kpi)

        gap[0]=np.abs(H_kpi[1]-H_kpi[0])
        gap[1]=np.abs(H_k0[2]-H_k0[1])
        gap[2]=np.abs(H_kpi[3]-H_kpi[2])
        #gap=H_kpi[1]-H_kpi[0]
        self._gap=gap
        return gap




