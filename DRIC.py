#!/usr/bin/python

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sympy import *
from scipy.interpolate import interp1d
from scipy.integrate import ode
from scipy.sparse import coo_matrix, csc_matrix, linalg as sla
from scipy.optimize import root,brentq

class ResistiveInterchange():
    """
    This class defines the methods to compute the resistive interchange (RIC) growth rate and eigenfunction,
    Including heat and momentum dissipation, as well as both electron and ion diamagnetic effects

    The parameters are

    beta
    eta
    chiperp
    chipar
    compr (Gamma, but the physics related to these equations is not validated, so use it at your own risks)
    tau = Ti/Te
    m poloidal mode number (we use a float rather than an integer)
    n toroidal mode number (we use a float rather than an integer)
    N the resolution in the radial direction    
    epsilon the aspect ratio (does not play any role except chipar is nonzero)
    """
    
    def __init__(self,
                 filename=None,
                 name='RIC',
                 option=None,
                 beta=None,
                 eta=None,
                 nu=None,
                 chiperp=None,
                 chipar=None,
                 compr=0.,
                 di=None,
                 tau=None,
                 m=None,
                 n=None,
                 N=None,
                 epsilon=0.1666):
        """
        Tasks to do at initialization
        """
        
        if filename==None:
            self.name = name

            if option==None:
                print " "
                print " Would you like to define the pressure and iota profiles via the interpolation of profiles (option = 1)"
                print " In this case A_pressure is supposed to be a data on a radius defined by "
                print " r = numpy.linspace(0,1,len(A_pressure))"
                print " Same for A_iota"
                print " The pressure is automaticaly normalized by dividing by max(A_pressure)"
                print " "
                print " Or via a series of coefficients (option = 2) ?"
                print " In this case A_pressure[0] and A_iota[0] are assumed to be the coefficients of higher order"
                print " "
                print " Or via symbolic functions (option = 3) ?"
                print " In this case, A_pressure and A_iota are symbolic functions of sympy.Symbol('r')"
                print " (pressure must be normalized in this case !)"
                
                print " "
                
                self.option = raw_input(" option (default 1) = ")
                
                if self.option == '':
                    self.option = 1
                else:
                    self.option = int(self.option)
            else:
                self.option=option

            if beta==None:
                self.beta = raw_input(" beta (default 0.05) = ")
                if self.beta=='':
                    self.beta=0.05
                else:
                    self.beta = float(self.beta)
            else:
                self.beta = beta
                
            if eta==None:
                self.eta = raw_input(" eta = (default 0.) ")
                if self.eta=='':
                    self.eta=0.
                else:
                    self.eta = float(self.eta)
            else:
                self.eta = eta
                
            if nu==None:
                self.nu = raw_input(" nu = (default 0.) ")
                if self.nu=='':
                    self.nu=0.
                else:
                    self.nu = float(self.nu)
            else:
                self.nu = nu

            if chiperp==None:
                self.chiperp = raw_input(" chiperp = (default 0.) ")
                if self.chiperp=='':
                    self.chiperp=0.
                else:
                    self.chiperp = float(self.chiperp)
            else:
                self.chiperp = chiperp

            if chipar==None:
                self.chipar = raw_input(" chipar = (default 0.) ")
                if self.chipar=='':
                    self.chipar=0.
                else:
                    self.chipar = float(self.chipar)
            else:
                self.chipar = chipar

            self.compr = compr
        
            if di==None:
                self.di = raw_input(" di = (default 0.) ")
                if self.di=='':
                    self.di=0.
                else:
                    self.di = float(self.di)
            else:
                self.di = di

            if tau==None:
                self.tau = raw_input(" tau = (default 1.) ")
                if self.tau=='':
                    self.tau=1.
                else:
                    self.tau = float(self.tau)
            else:
                self.tau = tau
        
            if m==None:
                self.m = raw_input(" m = (default 1.) ")
                if self.m=='':
                    self.m=1.
                else:
                    self.m = float(self.m)
            else:
                self.m = m
                
            if n==None:
                self.n = raw_input(" n = (default 1.) ")
                if self.n=='':
                    self.n=1.
                else:
                    self.n = float(self.n)
            else:
                self.n = n

            if N==None:
                self.N = raw_input(" N = (default 999) ")
                if self.N=='':
                    self.N=999
                else:
                    self.N = int(self.N)
            else:
                self.N = N
                
            self.epsilon = epsilon
            
            self.defined = False
            print " "
            print " Now you must define the profiles with method DefinePressureIotaProfiles"
            
            self.HasOneSolution = False
            self.solutions = []
        
            self._parameter_counts = []
            self._parameter_names = []

        else:
            # if filename!=None, load the data in filename
            self._Load(filename)

    def DefinePressureIotaProfiles(self,A_pressure,A_iota,rad_pressure=None,rad_iota=None):
        """
        Arguments : A_pressure, A_iota
        Each is a list or array of floats or integers which define the pressure and rotational transform profiles
        according to the following rules
        
        Three options are available:
        1 --> Through interpolation of input profiles 
        In this case A_pressure is supposed to be a data on a radius defined by 
        r = numpy.linspace(0,1,len(A_pressure))
        unless rad!=None
        Same for A_iota
        The pressure is automaticaly normalized by dividing by max(A_pressure)
       
        2 --> Through polynomial coefficients
        In this case A_pressure[0] and A_iota[0] are assumed to be the coefficients of higher order

        3 --> A_pressure and A_iota are symbolic functions of sympy.Symbol('r')
        (pressure must be normalized in this case !)
        """

        # save the parameters so they can be saved and everything can be reconstructed
        self.A_pressure = A_pressure
        self.A_iota = A_iota
        
        # Properties of the LHD
        # Field periods
        N_hel = 10.
        # Pole number
        l=2.

        if self.option==1:
            # If data is provided as list, transform to arrays
            if type(A_pressure)==list:
                A_pressure = np.array(A_pressure).astype(float)
            if type(A_iota)==list:
                A_iota = np.array(A_iota).astype(float)

            if rad_pressure==None:
                rad_pressure = np.linspace(0,1,len(A_pressure))
            else:
                if type(rad_pressure)==list:
                    rad_pressure=np.array(rad_pressure)
            self.rad_pressure = rad_pressure
            x = np.linspace(0,1,500)
            self.pressure = interp1d(rad_pressure,A_pressure/A_pressure.max(),kind='cubic')
            pressure_profile = self.pressure(x)
            pressure_prime_profile = np.gradient(pressure_profile,x[1]-x[0],edge_order=2)
            self.pressure_prime = interp1d(x,pressure_prime_profile,kind='cubic')

            if rad_iota==None:
                rad_iota = np.linspace(0,1,len(A_iota))
            else:
                if type(rad_iota)==list:
                    rad_iota=np.array(rad_iota)
            self.rad_iota = rad_iota

            self.iota = interp1d(rad_iota,A_iota,kind='cubic')
            iota_prime_profile = np.gradient(self.iota(x),x[1]-x[0],edge_order=2)

            Omega_prime_profile = 0.5*N_hel*(4*x*self.iota(x) + x**2*iota_prime_profile)/l
            self.Omega_prime = interp1d(x,Omega_prime_profile,kind='cubic')

        elif self.option==2:
            pressure_symb = self._polynomial(A_pressure)
            iota_symb = self._polynomial(A_iota)

            r = Symbol('r')
            self.pressure = lambdify(r,pressure_symb,'numpy')
            pressure_prime_symb = pressure_symb.diff(r)
            self.pressure_prime = lambdify(r,pressure_prime_symb,'numpy')
            self.iota = lambdify(r,iota_symb,'numpy')
            iota_prime_symb = iota_symb.diff(r)
            Omega_prime_symb = 0.5*N_hel*(4*r*iota_symb + r**2*iota_prime_symb)/l
            self.Omega_prime = lambdify(r,Omega_prime_symb,'numpy')

            self.rad_pressure = None
            self.rad_iota = None

        elif self.option==3:
            r = Symbol('r')
            self.pressure = lambdify(r,A_pressure,'numpy')
            pressure_prime_symb = A_pressure.diff(r)
            self.pressure_prime = lambdify(r,pressure_prime_symb,'numpy')
            self.iota = lambdify(r,A_iota,'numpy')
            iota_prime_symb = A_iota.diff(r)
            Omega_prime_symb = 0.5*N_hel*(4*r*A_iota + r**2*iota_prime_symb)/l
            self.Omega_prime = lambdify(r,Omega_prime_symb,'numpy')

            self.rad_pressure = None
            self.rad_iota = None

        self.X,self.dx = np.linspace(0,1,self.N+2,retstep=True)
        # The previous x above is no longer needed
        self.x = self.X[1:self.N+2]

        self.defined = True
        self._DefineSecondaryProfiles()

    def _polynomial(self,p):
        """
        Defines the polynomial P(X) = p[0]X^(n-1) + p[1]X^(n-2) + p[n-1], where n=len(p), in a symbolic way
        """

        r = Symbol('r')
        
        n = len(p)
        
        Pol = p[0]*r**(n-1)
        for i in range(1,n):
            Pol = Pol + p[i]*r**(n-1-i)

        return Pol

    def _DefineSecondaryProfiles(self):
        """
        Define the profiles which depend on the parameters, such as k_parallel, wstar, etc
        This routine has to be called EVERY TIME a parameter is changed !
        """

        # kparallel = m*iota - n
        self.nmi = self.m*self.iota(self.x) - self.n
         
        # Definition of wstari corresponding to the present normalization
        # Note that the normalized ion skin depth di stands for di/R0
        self.wstari = (- self.tau/(self.tau+1.)*0.5*self.beta/self.epsilon**2
                       *self.m*self.pressure_prime(self.x)/self.x*self.di)
        self.wstare = - self.wstari/self.tau

        self.dx1 = 1./self.dx
        self.__f_i = 2*self.dx1**2 + self.m**2/self.x**2
        self.__f_ip1 = self.dx1**2 + 0.5*self.dx1/self.x
        self.__f_im1 = self.dx1**2 - 0.5*self.dx1/self.x

    def _cont_2_ind(self,x_cont,x,tol=1e-2):
        """
        Finds the index ind such that x[ind] is closest to x_cont
        """
        if type(x)!=np.ndarray:
            x=np.array(x)

        x = x-x_cont
        ind=0
        sgn0=np.sign(x[0])
        sgn=sgn0
        
        try:
            while (sgn==sgn0)|(sgn==0):
                ind=ind+1
                sgn=np.sign(x[ind])
        except IndexError:
            print 'The right side of the array has been reached'
            print 'The last element of the list is returned'
            err=np.abs((x[-1])/x_cont) # Remember that we already subtracted x_cont from x
            ind=len(x)
            if err>tol:
                print 'The error is larger than the tolerance'
                print 'I returned an index but the value is not there in the array'
            else:
                print 'The error is less than the tolerance'
        # In the general case, we want the index to the left, hence subtract 1
        # If the exception is reached, we want the index of the last element, which is
        # len(x)-1, hence subtract 1 also in this case
        ind=ind-1
        
        return ind

    def _ResetParameters(self,beta=None,eta=None,nu=None,chiperp=None,chipar=None,compr=None,di=None,tau=None,m=None,n=None,N=None,epsilon=None):
        """
        Resets the parameters
        beta
        eta
        chiperp
        chipar
        compr (Gamma, but the physics related to these equations is not validated, so use it at your own risks)
        tau = Ti/Te
        m poloidal mode number (we use a float rather than an integer)
        n toroidal mode number (we use a float rather than an integer)
        N the resolution in the radial direction
        """

        if beta!=None:
            self.beta = beta

        if eta!=None:
            self.eta = eta

        if nu!=None:
            self.nu = nu

        if chiperp!=None:
            self.chiperp = chiperp

        if chipar!=None:
            self.chipar = chipar

        if compr!=None:
            self.compr = compr
        
        if di!=None:
            self.di = di

        if tau!=None:
            self.tau = tau
        
        if m!=None:
            self.m = m

        if n!=None:
            self.n = n

        if N!=None:
            self.N = N

        if epsilon!=None:
            self.epsilon = epsilon

        self._DefineSecondaryProfiles()
            
    def _Complete_Function_Set_Case_Psi_eq_1(self,omega):
        """
        Define all the functions to compute the matrix coefficiens of the complete complex problem
        Additional effects present:
        - resistivity
        - perpendicular viscosity
        - perpendicular heat diffusivity
        - parallel heat diffusivity
        - ion diamagnetic effects
        - electron diamagnetic effects
        """

        # Psi equation
        C_psi_psi_i = 1j*(omega-self.wstare) - self.eta*self.__f_i
        C_psi_psi_ip1 = self.eta*self.__f_ip1
        C_psi_psi_im1 = self.eta*self.__f_im1
        
        # To be consistent with chiperp, we must keep the (electon) pressure total and
        # have an term for phi and a term for p as follows
        C_psi_phi_i = -self.nmi
        C_psi_p_i = -1j*self.nmi[:-1]*self.wstare[:-1]*self.x[:-1]/(self.m*self.pressure_prime(self.x[:-1]))
        
        # Phi equation
        C_phi_phi_i = -1j*(omega-self.wstari)*self.__f_i
        C_phi_phi_ip1 = 1j*(omega-self.wstari)*self.__f_ip1
        C_phi_phi_im1 = 1j*(omega-self.wstari)*self.__f_im1
        
        C_phi_psi_i = -self.nmi*self.__f_i
        C_phi_psi_ip1 = self.nmi*self.__f_ip1
        C_phi_psi_im1 = self.nmi*self.__f_im1
        
        C_phi_p_i = self.m*self.beta*self.Omega_prime(self.x)/self.x
        
        C_phi_V_i = -self.nu*self.__f_i
        C_phi_V_ip1= self.nu*self.__f_ip1
        C_phi_V_im1 = self.nu*self.__f_im1

        # pressure equation
        C_p_p_i = 1j*(omega - self.beta*0.5*self.compr*self.pressure(self.x)*self.nmi**2/omega) - self.chiperp*self.__f_i - self.chipar*(self.nmi*self.epsilon)**2
        C_p_p_ip1 = self.chiperp*self.__f_ip1
        C_p_p_im1 = self.chiperp*self.__f_im1
    
        C_p_phi_i = self.m/self.x*self.pressure_prime(self.x)
        
        C_p_psi_i = -self.chipar*self.m*self.pressure_prime(self.x)/self.x*self.nmi*self.epsilon**2
        
        # V equation
        C_V_V_i = 1.*self.x**0
        C_V_phi_i = self.__f_i
        C_V_phi_ip1 = -self.__f_ip1
        C_V_phi_im1 = -self.__f_im1

        
        # Return arrays
        C_psi = [C_psi_psi_i,C_psi_psi_ip1,C_psi_psi_im1,
                 C_psi_phi_i,
                 C_psi_p_i]
        
        C_phi = [C_phi_phi_i,C_phi_phi_ip1,C_phi_phi_im1,
                 C_phi_psi_i,C_phi_psi_ip1,C_phi_psi_im1,
                 C_phi_p_i,
                 C_phi_V_i,C_phi_V_ip1,C_phi_V_im1]
        
        C_p   = [C_p_p_i,C_p_p_ip1,C_p_p_im1,
                 C_p_phi_i,
                 C_p_psi_i]
        
        C_V   = [C_V_V_i,
                 C_V_phi_i,C_V_phi_ip1,C_V_phi_im1]
        
        return C_psi,C_phi,C_p,C_V

    def _Sparse_Matrix_Psi_Case_Psi_eq_1(self,row,col,data,i,
                                         C_psi):
        """
        Defines the part of the matrix corresponding to the psi equation
        See the documentation
        
        Case_Psi_eq_1 means the following :
        The resistivity is non zero and the algorithm is set so that the non-vanishing boundary condition is
        on Psi and it is obtained in the first equation. The variables are thus
        
        (psi_0,...,psi_(N-1),psi_N,phi_0,...,phi_(N-1),p_0,...,p_(N-1),V_0,...,V_(N-1))
        
        and recall that x_-1 = 0, x_0 = dx, x_(N-1) = 1-dx, X_N = 1
        """
        
        N = self.N
        
        # Self coupling
        C_psi_i    =  C_psi[0]
        C_psi_ip1  =  C_psi[1]
        C_psi_im1  =  C_psi[2]
        
        # Coupling to phi
        C_phi_i    =  C_psi[3]
        
        # Coupling to p
        C_p_i = C_psi[4]
        
        def put(k,l,val):
            row.append(k)
            col.append(l)
            data.append(val)
            return row,col,data

        if i==0:
            row,col,data=put(i,i+N+1,C_phi_i[i])
            row,col,data=put(i,i+2*N+1,C_p_i[i])
            row,col,data=put(i,i+1,C_psi_ip1[i])
        elif i==N:
            row,col,data=put(i,i-1,C_psi_im1[i]+C_psi_ip1[i])
        else:
            row,col,data=put(i,i+N+1,C_phi_i[i])
            row,col,data=put(i,i+2*N+1,C_p_i[i])
            row,col,data=put(i,i-1,C_psi_im1[i])
            row,col,data=put(i,i+1,C_psi_ip1[i])
        row,col,data=put(i,i,C_psi_i[i])

        return row,col,data

    def _Sparse_Matrix_Phi_Case_Psi_eq_1(self,row,col,data,i,
                                         C_phi):
        """
        Defines the part of the matrix corresponding to the vorticity equation
        See the documentation
        
        Case_Psi_eq_1 means the following :
        The resistivity is non zero and the algorithm is set so that the non-vanishing boundary condition is
        on Psi and it is obtained in the first equation. The variables are thus
        
        (psi_0,...,psi_(N-1),psi_N,phi_0,...,phi_(N-1),p_0,...,p_(N-1),V_0,...,V_(N-1))
        
        and recall that x_-1 = 0, x_0 = dx, x_(N-1) = 1-dx, X_N = 1
        """

        N = self.N
        
        # Self coupling
        C_phi_i    =  C_phi[0]
        C_phi_ip1  =  C_phi[1]
        C_phi_im1  =  C_phi[2]
    
        # Coupling to psi
        C_psi_i    =  C_phi[3]
        C_psi_ip1  =  C_phi[4]
        C_psi_im1  =  C_phi[5]
        
        # Coupling to p
        C_p_i    =  C_phi[6]
        
        # Coupling to V the vorticity
        C_V_i    =  C_phi[7]
        C_V_ip1  =  C_phi[8]
        C_V_im1  =  C_phi[9]
        
        def put(k,l,val):
            row.append(k)
            col.append(l)
            data.append(val)
            return row,col,data
            
        if i==0:
            # phi (self)
            row,col,data=put(i+N+1,i+N+2,C_phi_ip1[i])
            # psi
            row,col,data=put(i+N+1,i+1,C_psi_ip1[i])
            # V
            row,col,data=put(i+N+1,i+3*N+2,C_V_ip1[i])        
        elif i==N-1:
            # phi (self)
            row,col,data=put(i+N+1,i+N,C_phi_im1[i])
            # psi
            row,col,data=put(i+N+1,i-1,C_psi_im1[i])
            row,col,data=put(i+N+1,i+1,C_psi_ip1[i])
            # V
            row,col,data=put(i+N+1,i+3*N,C_V_im1[i])        
        else:
            # phi (self)
            row,col,data=put(i+N+1,i+N+2,C_phi_ip1[i])
            row,col,data=put(i+N+1,i+N,C_phi_im1[i])
            # psi
            row,col,data=put(i+N+1,i+1,C_psi_ip1[i])
            row,col,data=put(i+N+1,i-1,C_psi_im1[i])
            # V
            row,col,data=put(i+N+1,i+3*N+2,C_V_ip1[i])
            row,col,data=put(i+N+1,i+3*N,C_V_im1[i])
        # phi (self)
        row,col,data=put(i+N+1,i+N+1,C_phi_i[i])
        # psi
        row,col,data=put(i+N+1,i,C_psi_i[i])
        # p
        row,col,data=put(i+N+1,i+2*N+1,C_p_i[i])
        # V
        row,col,data=put(i+N+1,i+3*N+1,C_V_i[i])

        return row,col,data

    def _Sparse_Matrix_P_Case_Psi_eq_1(self,row,col,data,i,
                                       C_p):
        """
        Defines the part of the matrix corresponding to the pressure equation
        See the documentation
        
        Case_Psi_eq_1 means the following :
        The resistivity is non zero and the algorithm is set so that the non-vanishing boundary condition is
        on Psi and it is obtained in the first equation. The variables are thus
        
        (psi_0,...,psi_(N-1),psi_N,phi_0,...,phi_(N-1),p_0,...,p_(N-1),V_0,...,V_(N-1))
        
        and recall that x_-1 = 0, x_0 = dx, x_(N-1) = 1-dx, X_N = 1
        """

        N = self.N
        
        # Self coupling
        C_p_i    =  C_p[0]
        C_p_ip1  =  C_p[1]
        C_p_im1  =  C_p[2]
        
        # Coupling to phi
        C_phi_i    =  C_p[3]
        
        # Coupling to psi
        C_psi_i    =  C_p[4]
        
        def put(k,l,val):
            row.append(k)
            col.append(l)
            data.append(val)
            return row,col,data
            
        if i==0:
            row,col,data=put(i+2*N+1,i+2*N+2,C_p_ip1[i])
        elif i==N-1:
            row,col,data=put(i+2*N+1,i+2*N,C_p_im1[i])
        else:
            row,col,data=put(i+2*N+1,i+2*N+2,C_p_ip1[i])
            row,col,data=put(i+2*N+1,i+2*N,C_p_im1[i])
        row,col,data=put(i+2*N+1,i+2*N+1,C_p_i[i])
        row,col,data=put(i+2*N+1,i+N+1,C_phi_i[i])
        row,col,data=put(i+2*N+1,i,C_psi_i[i])
        
        return row,col,data

    def _Sparse_Matrix_V_Case_Psi_eq_1(self,row,col,data,i,
                                       C_V):
        """
        Defines the part of the matrix corresponding to the definition of vorticity lapla(phi) = V
        See the documentation
        
        Case_Psi_eq_1 means the following :
        The resistivity is non zero and the algorithm is set so that the non-vanishing boundary condition is
        on Psi and it is obtained in the first equation. The variables are thus
        
        (psi_0,...,psi_(N-1),psi_N,phi_0,...,phi_(N-1),p_0,...,p_(N-1),V_0,...,V_(N-1))
        
        and recall that x_-1 = 0, x_0 = dx, x_(N-1) = 1-dx, X_N = 1
        """
        
        N = self.N
        
        # Self coupling
        C_V_i    =  C_V[0]

        # Coupling to phi
        C_phi_i    =  C_V[1]
        C_phi_ip1  =  C_V[2]
        C_phi_im1  =  C_V[3]
        
        def put(k,l,val):
            row.append(k)
            col.append(l)
            data.append(val)
            return row,col,data
            
        if i==0:
            row,col,data=put(i+3*N+1,i+N+2,C_phi_ip1[i])
        elif i==N-1:
            row,col,data=put(i+3*N+1,i+N,C_phi_im1[i])
        else:
            row,col,data=put(i+3*N+1,i+N+2,C_phi_ip1[i])
            row,col,data=put(i+3*N+1,i+N,C_phi_im1[i])
        row,col,data=put(i+3*N+1,i+3*N+1,C_V_i[i])
        row,col,data=put(i+3*N+1,i+N+1,C_phi_i[i])
        
        return row,col,data


    def _Build_Matrix_Case_Psi_eq_1(self,C_psi,C_phi,C_p,C_V):
        """
        Build the matrix
        Case_Psi_eq_1 means the following :
        The resistivity is non zero and the algorithm is set so that the non-vanishing boundary condition is
        on Psi and it is obtained in the first equation. The variables are thus
        
        (psi_0,...,psi_(N-1),psi_N,phi_0,...,phi_(N-1),p_0,...,p_(N-1),V_0,...,V_(N-1))
        
        and recall that x_-1 = 0, x_0 = dx, x_(N-1) = 1-dx, X_N = 1
        """

        row=[]
        col=[]
        data=[]

        for i in range(self.N+1):
            row,col,data = self._Sparse_Matrix_Psi_Case_Psi_eq_1(row,col,data,i,
                                                           C_psi)
        for i in range(self.N):
            row,col,data = self._Sparse_Matrix_Phi_Case_Psi_eq_1(row,col,data,i,
                                                           C_phi)
        for i in range(self.N):
            row,col,data = self._Sparse_Matrix_P_Case_Psi_eq_1(row,col,data,i,
                                                         C_p)
        for i in range(self.N):
            row,col,data = self._Sparse_Matrix_V_Case_Psi_eq_1(row,col,data,i,
                                                         C_V)

        M = coo_matrix((data, (row, col)), shape=(4*self.N+1,4*self.N+1))

        return M

    def _RHS_Case_Psi_eq_1(self,C_psi,C_phi,C_p,C_V):
        """
        Define the vector B, RHS of AX=B
        
        Case_Psi_eq_1 means the following :
        The resistivity is non zero and the algorithm is set so that the non-vanishing boundary condition is
        on Psi and it is obtained in the first equation. The variables are thus
        
        (psi_0,...,psi_(N-1),psi_N,phi_0,...,phi_(N-1),p_0,...,p_(N-1),V_0,...,V_(N-1))
        
        and recall that x_-1 = 0, x_0 = dx, x_(N-1) = 1-dx, X_N = 1
        
        In theory we should have many terms corresponding to all the boundary conditions
        In practice, since all the boundary conditions are 0, we write only the boundary condition corresponding to
        psi'(r=1) = 1
        
        """

        C_psi_psi_ip1 = C_psi[1]

        B = np.zeros(4*self.N+1,dtype=complex)

        psi_prime_N = 1
        B[self.N] = -2*C_psi_psi_ip1[-1]*self.dx*psi_prime_N

        return B

    def _Problem_Solution_Case_Psi_eq_1(self,omega):
        """
        Build matrix
        LU decompose
        Build RHS
        Find SOL=(psi_0, ... ,psi_N,phi_0, ... ,phi_N-1,p_0, ... ,p_N-1,V_0, ... ,V_N-1)
        Return X, Psi, Phi, P, V
        
        Case_Psi_eq_1 means the following :
        The resistivity is non zero and the algorithm is set so that the non-vanishing boundary condition is
        on Psi and it is obtained in the first equation. The variables are thus
        
        (psi_0,...,psi_(N-1),psi_N,phi_0,...,phi_(N-1),p_0,...,p_(N-1),V_0,...,V_(N-1))
        
        and recall that x_-1 = 0, x_0 = dx, x_(N-1) = 1-dx, X_N = 1
        """

        N = self.N

        C_psi,C_phi,C_p,C_V = self._Complete_Function_Set_Case_Psi_eq_1(omega)

        A = self._Build_Matrix_Case_Psi_eq_1(C_psi,C_phi,C_p,C_V)

        LU = sla.splu(csc_matrix(A))
    
        B = self._RHS_Case_Psi_eq_1(C_psi,C_phi,C_p,C_V)

        # SOL = sla.bicg(csc_matrix(A),B)

        SOL = LU.solve(B)

        psi_m1 = 0
        phi_m1 = 0
        phi_N = 0
        p_m1 = 0
        p_N = 0
        V_m1 = 0
        V_N = 0

        Psi=np.zeros(N+2,dtype=complex)
        Psi[0]=psi_m1
        Psi[1:N+2]=SOL[0:N+1]
        
        Phi=np.zeros(N+2,dtype=complex)
        Phi[0]=phi_m1
        Phi[1:N+1]=SOL[N+1:2*N+1]
        Phi[N+1]=phi_N
        
        P=np.zeros(N+2,dtype=complex)
        P[0]=p_m1
        P[1:N+1]=SOL[2*N+1:3*N+1]
        P[N+1]=p_N
        
        V=np.zeros(N+2,dtype=complex)
        V[0]=V_m1
        V[1:N+1]=SOL[3*N+1:]
        V[N+1]=V_N
        
        return Psi,Phi,P,V

    def _Complete_Function_Set_Case_Psi_eq_2(self,omega):
        """
        Define all the functions to compute the matrix coefficiens of the complete complex problem
        Additional effects present:
        - perpendicular heat diffusivity
        - parallel heat diffusivity
        - ion diamagnetic effects
        - electron diamagnetic effects
        """

        # Psi equation
        C_psi_psi_i = 1j*(omega-self.wstare) - self.eta*self.__f_i
        C_psi_psi_ip1 = self.eta*self.__f_ip1
        C_psi_psi_im1 = self.eta*self.__f_im1
        
        # Case which is not consistent with chiperp
        #C_psi_phi_i = -nmi*(omega-wstare)/omega
        
        # To be consistent with chiperp, we must keep the (electon) pressure total and
        # have an term for phi and a term for p as follows
        C_psi_phi_i = -self.nmi
        C_psi_p_i = -1j*self.nmi[:-1]*self.wstare[:-1]*self.x[:-1]/(self.m*self.pressure(self.x[:-1]))
        
        # Phi equation
        C_phi_psi_i = -self.nmi*self.__f_i
        C_phi_psi_ip1 = self.nmi*self.__f_ip1
        C_phi_psi_im1 = self.nmi*self.__f_im1
        
        C_phi_p_i = self.m*self.beta*self.Omega_prime(self.x)/self.x
        
        C_phi_V_i = 1j*(omega-self.wstari)
        
        # pressure equation
        C_p_p_i = 1j*(omega - self.beta*0.5*self.compr*self.pressure(self.x)*self.nmi**2/omega) - self.chiperp*self.__f_i - self.chipar*(self.nmi*self.epsilon)**2
        C_p_p_ip1 = self.chiperp*self.__f_ip1
        C_p_p_im1 = self.chiperp*self.__f_im1
        
        C_p_phi_i = self.m/self.x*self.pressure_prime(self.x)
        
        C_p_psi_i = -self.chipar*self.m*self.pressure_prime(self.x)/self.x*self.nmi*self.epsilon**2    
        
        # V equation
        C_V_V_i = 1.*self.x**0
        C_V_phi_i = self.__f_i
        C_V_phi_ip1 = -self.__f_ip1
        C_V_phi_im1 = -self.__f_im1
        
        # Return functions
        C_psi = [C_psi_psi_i,C_psi_psi_ip1,C_psi_psi_im1,
                 C_psi_phi_i,
                 C_psi_p_i]
        
        C_phi = [C_phi_psi_i,C_phi_psi_ip1,C_phi_psi_im1,
                 C_phi_p_i,
                 C_phi_V_i]
        
        C_p   = [C_p_p_i,C_p_p_ip1,C_p_p_im1,
                 C_p_phi_i,
                 C_p_psi_i]
        
        C_V   = [C_V_V_i,
                 C_V_phi_i,C_V_phi_ip1,C_V_phi_im1]
        
        return C_psi,C_phi,C_p,C_V

    def _Sparse_Matrix_Psi_Case_Psi_eq_2(self,row,col,data,i,
                                         C_psi):
        """
        Defines the part of the matrix corresponding to the psi equation
        See the documentation
        
        Case_Psi_eq_2 means the following :
        The resistivity and the viscosity are zero and the algorithm is set so that the non-vanishing boundary condition is
        on Psi and it is obtained in the second equation. The variables are thus
        
        (phi_0,...,phi_(N-1),psi_0,...,psi_(N-1),psi_N,p_0,...,p_(N-1),V_0,...,V_(N-1))
        
        and recall that x_-1 = 0, x_0 = dx, x_(N-1) = 1-dx, X_N = 1
        
        In this case the equation is purely algebraic
        """
        N = self.N

        # Self coupling
        C_psi_i    =  C_psi[0]
        C_psi_ip1  =  C_psi[1]
        C_psi_im1  =  C_psi[2]
        
        # Coupling to phi
        C_phi_i    =  C_psi[3]
        
        # Coupling to p
        C_p_i = C_psi[4]
        
        def put(k,l,val):
            row.append(k)
            col.append(l)
            data.append(val)
            return row,col,data
            
        # psi (self)
        row,col,data=put(i,i+N,C_psi_i[i])
        # phi
        row,col,data=put(i,i,C_phi_i[i])
        # p
        row,col,data=put(i,i+2*N+1,C_p_i[i])
        
        return row,col,data

    def _Sparse_Matrix_Phi_Case_Psi_eq_2(self,row,col,data,i,
                                         C_phi):
        """
        Defines the part of the matrix corresponding to the vorticity equation
        See the documentation
        
        Case_Psi_eq_2 means the following :
        The resistivity and the viscosity are zero and the algorithm is set so that the non-vanishing boundary condition is
        on Psi and it is obtained in the second equation. The variables are thus
        
        (phi_0,...,phi_(N-1),psi_0,...,psi_(N-1),psi_N,p_0,...,p_(N-1),V_0,...,V_(N-1))
        
        and recall that x_-1 = 0, x_0 = dx, x_(N-1) = 1-dx, X_N = 1
        
        We rewrite the vorticity term as V instead of lapl(phi) so phi does not directly appear here
        """

        N = self.N

        # Coupling to psi
        C_psi_i    =  C_phi[0]
        C_psi_ip1  =  C_phi[1]
        C_psi_im1  =  C_phi[2]
        
        # Coupling to p
        C_p_i    =  C_phi[3]
        
        # Coupling to V the vorticity
        C_V_i    =  C_phi[4]
        
        def put(k,l,val):
            row.append(k)
            col.append(l)
            data.append(val)
            return row,col,data
            
        if i==0:
            # psi
            row,col,data=put(i+N,i+N+1,C_psi_ip1[i])
            # p
            row,col,data=put(i+N,i+2*N+1,C_p_i[i])
            # V
            row,col,data=put(i+N,i+3*N+1,C_V_i[i])
        elif i==N:
            # psi
            row,col,data=put(i+N,i+N-1,C_psi_im1[i]+C_psi_ip1[i])
        else:
            # psi
            row,col,data=put(i+N,i+N+1,C_psi_ip1[i])
            row,col,data=put(i+N,i+N-1,C_psi_im1[i])
            # p
            row,col,data=put(i+N,i+2*N+1,C_p_i[i])
            # V
            row,col,data=put(i+N,i+3*N+1,C_V_i[i])
        # psi
        row,col,data=put(i+N,i+N,C_psi_i[i])
        
        return row,col,data

    def _Sparse_Matrix_P_Case_Psi_eq_2(self,row,col,data,i,
                                       C_p):
        """
        Defines the part of the matrix corresponding to the pressure equation
        See the documentation
        
        Case_Psi_eq_2 means the following :
        The resistivity and the viscosity are zero and the algorithm is set so that the non-vanishing boundary condition is
        on Psi and it is obtained in the second equation. The variables are thus
        
        (phi_0,...,phi_(N-1),psi_0,...,psi_(N-1),psi_N,p_0,...,p_(N-1),V_0,...,V_(N-1))
        
        and recall that x_-1 = 0, x_0 = dx, x_(N-1) = 1-dx, X_N = 1
        """

        N = self.N

        # Self coupling
        C_p_i    =  C_p[0]
        C_p_ip1  =  C_p[1]
        C_p_im1  =  C_p[2]
        
        # Coupling to phi
        C_phi_i    =  C_p[3]
        
        # Coupling to psi
        C_psi_i    =  C_p[4]
        
        def put(k,l,val):
            row.append(k)
            col.append(l)
            data.append(val)
            return row,col,data
            
        if i==0:
            row,col,data=put(i+2*N+1,i+2*N+2,C_p_ip1[i])
        elif i==N-1:
            row,col,data=put(i+2*N+1,i+2*N,C_p_im1[i])
        else:
            row,col,data=put(i+2*N+1,i+2*N+2,C_p_ip1[i])
            row,col,data=put(i+2*N+1,i+2*N,C_p_im1[i])
        row,col,data=put(i+2*N+1,i+2*N+1,C_p_i[i])
        row,col,data=put(i+2*N+1,i,C_phi_i[i])
        row,col,data=put(i+2*N+1,i+N,C_psi_i[i])

        return row,col,data

    def _Sparse_Matrix_V_Case_Psi_eq_2(self,row,col,data,i,
                                       C_V):
        """
        Defines the part of the matrix corresponding to the definition of vorticity lapla(phi) = V
        See the documentation
        
        Case_Psi_eq_2 means the following :
        The resistivity and the viscosity are zero and the algorithm is set so that the non-vanishing boundary condition is
        on Psi and it is obtained in the second equation. The variables are thus
        
        (phi_0,...,phi_(N-1),psi_0,...,psi_(N-1),psi_N,p_0,...,p_(N-1),V_0,...,V_(N-1))
        
        and recall that x_-1 = 0, x_0 = dx, x_(N-1) = 1-dx, X_N = 1
        """

        N = self.N

        # Self coupling
        C_V_i    =  C_V[0]
        
        # Coupling to phi
        C_phi_i    =  C_V[1]
        C_phi_ip1  =  C_V[2]
        C_phi_im1  =  C_V[3]
        
        def put(k,l,val):
            row.append(k)
            col.append(l)
            data.append(val)
            return row,col,data
            
        if i==0:
            row,col,data=put(i+3*N+1,i+1,C_phi_ip1[i])
        elif i==N-1:
            row,col,data=put(i+3*N+1,i-1,C_phi_im1[i])
        else:
            row,col,data=put(i+3*N+1,i+1,C_phi_ip1[i])
            row,col,data=put(i+3*N+1,i-1,C_phi_im1[i])
        row,col,data=put(i+3*N+1,i+3*N+1,C_V_i[i])
        row,col,data=put(i+3*N+1,i,C_phi_i[i])
        
        return row,col,data

    def _Build_Matrix_Case_Psi_eq_2(self,C_psi,C_phi,C_p,C_V):
        """
        Build the matrix
        Case_Psi_eq_2 means the following :
        The resistivity and the viscosity are zero and the algorithm is set so that the non-vanishing boundary condition is
        on Psi and it is obtained in the second equation. The variables are thus
        
        (phi_0,...,phi_(N-1),psi_0,...,psi_(N-1),psi_N,p_0,...,p_(N-1),V_0,...,V_(N-1))
        
        and recall that x_-1 = 0, x_0 = dx, x_(N-1) = 1-dx, X_N = 1
        """
        
        row=[]
        col=[]
        data=[]

        for i in range(self.N):
            row,col,data = self._Sparse_Matrix_Psi_Case_Psi_eq_2(row,col,data,i,
                                                           C_psi)
        for i in range(self.N+1):
            row,col,data = self._Sparse_Matrix_Phi_Case_Psi_eq_2(row,col,data,i,
                                                                 C_phi)
        for i in range(self.N):
            row,col,data = self._Sparse_Matrix_P_Case_Psi_eq_2(row,col,data,i,
                                                               C_p)
        for i in range(self.N):
            row,col,data = self._Sparse_Matrix_V_Case_Psi_eq_2(row,col,data,i,
                                                               C_V)

        M = coo_matrix((data, (row, col)), shape=(4*self.N+1,4*self.N+1))

        return M

    def _RHS_Case_Psi_eq_2(self,C_psi,C_phi,C_p,C_V):
        """
        Define the vector B, RHS of AX=B
        
        Case_Psi_eq_2 means the following :
        The resistivity and the viscosity are zero and the algorithm is set so that the non-vanishing boundary condition is
        on Psi and it is obtained in the second equation. The variables are thus
        
        (phi_0,...,phi_(N-1),psi_0,...,psi_(N-1),psi_N,p_0,...,p_(N-1),V_0,...,V_(N-1))
        
        and recall that x_-1 = 0, x_0 = dx, x_(N-1) = 1-dx, X_N = 1
        
        In theory we should have many terms corresponding to all the boundary conditions
        In practice, since all the boundary conditions are 0, we write only the boundary condition corresponding to
        psi'(r=1) = 1
        
        """
        
        C_phi_psi_ip1 = C_phi[1]
        
        B = np.zeros(4*self.N+1,dtype=complex)
        
        psi_prime_N = 1
        B[2*self.N] = -2*C_phi_psi_ip1[-1]*self.dx*psi_prime_N
        
        return B

    def _Problem_Solution_Case_Psi_eq_2(self,omega):
        """
        Build matrix
        LU decompose
        Build RHS
        Find SOL=(psi_0, ... ,psi_N,phi_0, ... ,phi_N-1,p_0, ... ,p_N-1,V_0, ... ,V_N-1)
        Return X, Psi, Phi, P, V
        
        Case_Psi_eq_2 means the following :
        The resistivity and the viscosity are zero and the algorithm is set so that the non-vanishing boundary condition is
        on Psi and it is obtained in the second equation. The variables are thus
        
        (phi_0,...,phi_(N-1),psi_0,...,psi_(N-1),psi_N,p_0,...,p_(N-1),V_0,...,V_(N-1))
        
        and recall that x_-1 = 0, x_0 = dx, x_(N-1) = 1-dx, X_N = 1
        """

        N = self.N

        C_psi,C_phi,C_p,C_V = self._Complete_Function_Set_Case_Psi_eq_2(omega)
    
        A = self._Build_Matrix_Case_Psi_eq_2(C_psi,C_phi,C_p,C_V)

        LU = sla.splu(csc_matrix(A))
    
        B = self._RHS_Case_Psi_eq_2(C_psi,C_phi,C_p,C_V)

        SOL = LU.solve(B)
        
        psi_m1 = 0
        phi_m1 = 0
        phi_N = 0
        p_m1 = 0
        p_N = 0
        V_m1 = 0
        V_N = 0
        
        Phi=np.zeros(N+2,dtype=complex)
        Phi[0]=phi_m1
        Phi[1:N+1]=SOL[0:N]
        Phi[N+1]=phi_N
        
        Psi=np.zeros(N+2,dtype=complex)
        Psi[0]=psi_m1
        Psi[1:N+2]=SOL[N:2*N+1]
        
        P=np.zeros(N+2,dtype=complex)
        P[0]=p_m1
        P[1:N+1]=SOL[2*N+1:3*N+1]
        P[N+1]=p_N
        
        V=np.zeros(N+2,dtype=complex)
        V[0]=V_m1
        V[1:N+1]=SOL[3*N+1:]
        V[N+1]=V_N
        
        return Psi,Phi,P,V

    def _Complete_Function_Set_Case_Phi_eq_4(self,omega):
        """
        Define all the functions to compute the matrix coefficiens of the complete complex problem
        Additional effects present:
        - perpendicular viscosity
        - parallel heat conductivity
        - perpendicular heat conductivity
        - ion diamagnetic effects
        - electron diamagnetic effects
        """
        # Psi equation
        C_psi_psi_i = 1j*(omega-self.wstare)
        
        # To be consistent with chiperp, we must keep the (electon) pressure total and
        # have a term for phi and a term for p as follows
        C_psi_phi_i = -self.nmi
        C_psi_p_i = -1j*self.nmi[:-1]*self.wstare[:-1]*self.x[:-1]/(self.m*self.pressure_prime(self.x[:-1]))
        
        # Phi equation
        C_phi_psi_i = -self.nmi*self.__f_i
        C_phi_psi_ip1 = self.nmi*self.__f_ip1
        C_phi_psi_im1 = self.nmi*self.__f_im1
        
        C_phi_p_i = self.m*self.beta*self.Omega_prime(self.x)/self.x
        
        C_phi_V_i = 1j*(omega-self.wstari) - self.nu*self.__f_i
        C_phi_V_ip1 = self.nu*self.__f_ip1
        C_phi_V_im1 = self.nu*self.__f_im1
        
        # pressure equation
        C_p_p_i = 1j*(omega - self.beta*0.5*self.compr*self.pressure(self.x)*self.nmi**2/omega) - self.chiperp*self.__f_i - self.chipar*(self.nmi*self.epsilon)**2
        C_p_p_ip1 = self.chiperp*self.__f_ip1
        C_p_p_im1 = self.chiperp*self.__f_im1
        
        C_p_phi_i = self.m/self.x*self.pressure_prime(self.x)
        
        # V equation
        C_V_V_i = 1.*self.x**0
        C_V_phi_i = self.__f_i
        C_V_phi_ip1 = -self.__f_ip1
        C_V_phi_im1 = -self.__f_im1
        
        C_p_psi_i = -self.chipar*self.m*self.pressure_prime(self.x)/self.x*self.nmi*self.epsilon**2
        
        # Return arrays
        C_psi = [C_psi_psi_i,
                 C_psi_phi_i,
                 C_psi_p_i]
        
        C_phi = [C_phi_psi_i,C_phi_psi_ip1,C_phi_psi_im1,
                 C_phi_p_i,
                 C_phi_V_i,C_phi_V_ip1,C_phi_V_im1]
        
        C_p   = [C_p_p_i,C_p_p_ip1,C_p_p_im1,
                 C_p_phi_i,
                 C_p_psi_i]
        
        C_V   = [C_V_V_i,
                 C_V_phi_i,C_V_phi_ip1,C_V_phi_im1]
        
        return C_psi,C_phi,C_p,C_V

    def _Sparse_Matrix_Psi_Case_Phi_eq_4(self,row,col,data,i,
                                         C_psi):
        """
        Defines the part of the matrix corresponding to the psi equation
        See the documentation
        
        Case_Phi_eq_4 means the following :
        The resistivity is zero but the viscosity is non zero
        The algorithm is set so that the non-vanishing boundary condition is
        on Phi and it is obtained in the fourth equation. The variables are
        
        (psi_0,...,psi_(N-1),phi_0,...,phi_(N-1),phi_N,p_0,...,p_(N-1),V_0,...,V_(N-1))
        
        and recall that x_-1 = 0, x_0 = dx, x_(N-1) = 1-dx, X_N = 1
        
        In this case the equation is purely algebraic
        """

        N = self.N

        # Self coupling
        C_psi_i    =  C_psi[0]
        
        # Coupling to phi
        C_phi_i    =  C_psi[1]
        
        # Coupling to p
        C_p_i = C_psi[2]
        
        def put(k,l,val):
            row.append(k)
            col.append(l)
            data.append(val)
            return row,col,data
            
        # psi (self)
        row,col,data=put(i,i,C_psi_i[i])
        # phi
        row,col,data=put(i,i+N,C_phi_i[i])
        # p
        row,col,data=put(i,i+2*N+1,C_p_i[i])
        
        return row,col,data

    def _Sparse_Matrix_Phi_Case_Phi_eq_4(self,row,col,data,i,
                                         C_phi):
        """
        Defines the part of the matrix corresponding to the vorticity equation
        See the documentation
        
        Case_Phi_eq_4 means the following :
        The resistivity is zero but the viscosity is non zero
        The algorithm is set so that the non-vanishing boundary condition is
        on Phi and it is obtained in the fourth equation. The variables are
        
        (psi_0,...,psi_(N-1),phi_0,...,phi_(N-1),phi_N,p_0,...,p_(N-1),V_0,...,V_(N-1))
        
        and recall that x_-1 = 0, x_0 = dx, x_(N-1) = 1-dx, X_N = 1
        """
        
        N = self.N

        # Coupling to psi
        C_psi_i    =  C_phi[0]
        C_psi_ip1  =  C_phi[1]
        C_psi_im1  =  C_phi[2]
        
        # Coupling to p
        C_p_i    =  C_phi[3]
        
        # Coupling to V the vorticity
        C_V_i    =  C_phi[4]
        C_V_ip1  =  C_phi[5]
        C_V_im1  =  C_phi[6]
        
        def put(k,l,val):
            row.append(k)
            col.append(l)
            data.append(val)
            return row,col,data

        if i==0:
            # psi
            row,col,data=put(i+N,i+1,C_psi_ip1[i])
            # V
            row,col,data=put(i+N,i+3*N+2,C_V_ip1[i])
        elif i==N-1:
            # psi
            row,col,data=put(i+N,i-1,C_psi_im1[i])
            # V
            row,col,data=put(i+N,i+3*N,C_V_im1[i])
        else:
            # psi
            row,col,data=put(i+N,i+1,C_psi_ip1[i])
            row,col,data=put(i+N,i-1,C_psi_im1[i])
            # V
            row,col,data=put(i+N,i+3*N+2,C_V_ip1[i])
            row,col,data=put(i+N,i+3*N,C_V_im1[i])
        # psi
        row,col,data=put(i+N,i,C_psi_i[i])
        # p
        row,col,data=put(i+N,i+2*N+1,C_p_i[i])
        # V
        row,col,data=put(i+N,i+3*N+1,C_V_i[i])

        return row,col,data

    def _Sparse_Matrix_P_Case_Phi_eq_4(self,row,col,data,i,
                                       C_p):
        """
        Defines the part of the matrix corresponding to the pressure equation
        See the documentation
        
        Case_Phi_eq_4 means the following :
        The resistivity is zero but the viscosity is non zero
        The algorithm is set so that the non-vanishing boundary condition is
        on Phi and it is obtained in the fourth equation. The variables are
        
        (psi_0,...,psi_(N-1),phi_0,...,phi_(N-1),phi_N,p_0,...,p_(N-1),V_0,...,V_(N-1))
        
        and recall that x_-1 = 0, x_0 = dx, x_(N-1) = 1-dx, X_N = 1
        """

        N = self.N

        # Self coupling
        C_p_i    =  C_p[0]
        C_p_ip1  =  C_p[1]
        C_p_im1  =  C_p[2]
        
        # Coupling to phi
        C_phi_i    =  C_p[3]
        
        # Coupling to psi
        C_psi_i    =  C_p[4]
        
        def put(k,l,val):
            row.append(k)
            col.append(l)
            data.append(val)
            return row,col,data
            
        if i==0:
            row,col,data=put(i+2*N,i+2*N+2,C_p_ip1[i])
        elif i==N-1:
            row,col,data=put(i+2*N,i+2*N,C_p_im1[i])
        else:
            row,col,data=put(i+2*N,i+2*N+2,C_p_ip1[i])
            row,col,data=put(i+2*N,i+2*N,C_p_im1[i])
        row,col,data=put(i+2*N,i+2*N+1,C_p_i[i])
        row,col,data=put(i+2*N,i+N,C_phi_i[i])
        row,col,data=put(i,i,C_psi_i[i])

        return row,col,data

    def _Sparse_Matrix_V_Case_Phi_eq_4(self,row,col,data,i,
                                       C_V):
        """
        Defines the part of the matrix corresponding to the definition of vorticity lapla(phi) = V
        See the documentation
        
        Case_Phi_eq_4 means the following :
        The resistivity is zero but the viscosity is non zero
        The algorithm is set so that the non-vanishing boundary condition is
        on Phi and it is obtained in the fourth equation. The variables are
        
        (psi_0,...,psi_(N-1),phi_0,...,phi_(N-1),phi_N,p_0,...,p_(N-1),V_0,...,V_(N-1))
        
        and recall that x_-1 = 0, x_0 = dx, x_(N-1) = 1-dx, X_N = 1
        """
        
        N = self.N

        # Self coupling
        C_V_i    =  C_V[0]
        
        # Coupling to phi
        C_phi_i    =  C_V[1]
        C_phi_ip1  =  C_V[2]
        C_phi_im1  =  C_V[3]
        
        def put(k,l,val):
            row.append(k)
            col.append(l)
            data.append(val)
            return row,col,data
            
        if i==0:
            row,col,data=put(i+3*N,i+N+1,C_phi_ip1[i])
            row,col,data=put(i+3*N,i+3*N+1,C_V_i[i])
        elif i==N:
            row,col,data=put(i+3*N,i+N-1,C_phi_im1[i]+C_phi_ip1[i])
        else:
            row,col,data=put(i+3*N,i+N+1,C_phi_ip1[i])
            row,col,data=put(i+3*N,i+N-1,C_phi_im1[i])
            row,col,data=put(i+3*N,i+3*N+1,C_V_i[i])
        row,col,data=put(i+3*N,i+N,C_phi_i[i])

        return row,col,data


    def _Build_Matrix_Case_Phi_eq_4(self,C_psi,C_phi,C_p,C_V):
        """
        Build the matrix
        
        Case_Phi_eq_4 means the following :
        The resistivity is zero but the viscosity is non zero
        The algorithm is set so that the non-vanishing boundary condition is
        on Phi and it is obtained in the fourth equation. The variables are
        
        (psi_0,...,psi_(N-1),phi_0,...,phi_(N-1),phi_N,p_0,...,p_(N-1),V_0,...,V_(N-1))
        
        and recall that x_-1 = 0, x_0 = dx, x_(N-1) = 1-dx, X_N = 1
        """
        
        row=[]
        col=[]
        data=[]
        
        for i in range(self.N):
            row,col,data = self._Sparse_Matrix_Psi_Case_Phi_eq_4(row,col,data,i,
                                                                 C_psi)
        for i in range(self.N):
            row,col,data = self._Sparse_Matrix_Phi_Case_Phi_eq_4(row,col,data,i,
                                                                 C_phi)
        for i in range(self.N):
            row,col,data = self._Sparse_Matrix_P_Case_Phi_eq_4(row,col,data,i,
                                                               C_p)
        for i in range(self.N+1):
            row,col,data = self._Sparse_Matrix_V_Case_Phi_eq_4(row,col,data,i,
                                                               C_V)

        M = coo_matrix((data, (row, col)), shape=(4*self.N+1,4*self.N+1))
        
        return M

    def _RHS_Case_Phi_eq_4(self,C_psi,C_phi,C_p,C_V):
        """
        Define the vector B, RHS of AX=B
        
        Case_Phi_eq_4 means the following :
        The resistivity is zero but the viscosity is non zero
        The algorithm is set so that the non-vanishing boundary condition is
        on Phi and it is obtained in the fourth equation. The variables are
        
        (psi_0,...,psi_(N-1),phi_0,...,phi_(N-1),phi_N,p_0,...,p_(N-1),V_0,...,V_(N-1))
        
        and recall that x_-1 = 0, x_0 = dx, x_(N-1) = 1-dx, X_N = 1
        
        In theory we should have many terms corresponding to all the boundary conditions
        In practice, since all the boundary conditions are 0, we write only the boundary condition corresponding to
        psi'(r=1) = 1
        """
    
        C_V_phi_ip1 = C_V[2]

        B = np.zeros(4*self.N+1,dtype=complex)

        phi_prime_N = 1
        B[4*self.N] = -2*C_V_phi_ip1[-1]*self.dx*phi_prime_N

        return B

    def _Problem_Solution_Case_Phi_eq_4(self,omega):
        """
        Build matrix
        LU decompose
        Build RHS
        Find SOL=(psi_0, ... ,psi_N,phi_0, ... ,phi_N-1,p_0, ... ,p_N-1,V_0, ... ,V_N-1)
        Return X, Psi, Phi, P, V
        
        Case_Phi_eq_4 means the following :
        The resistivity is zero but the viscosity is non zero
        The algorithm is set so that the non-vanishing boundary condition is
        on Phi and it is obtained in the fourth equation. The variables are
        
        (psi_0,...,psi_(N-1),phi_0,...,phi_(N-1),phi_N,p_0,...,p_(N-1),V_0,...,V_(N-1))
        
        and recall that x_-1 = 0, x_0 = dx, x_(N-1) = 1-dx, X_N = 1
        """

        N = self.N

        C_psi,C_phi,C_p,C_V = self._Complete_Function_Set_Case_Phi_eq_4(omega)

        A = self._Build_Matrix_Case_Phi_eq_4(C_psi,C_phi,C_p,C_V)

        LU = sla.splu(csc_matrix(A))
    
        B = self._RHS_Case_Phi_eq_4(C_psi,C_phi,C_p,C_V)
        
        SOL = LU.solve(B)
        
        psi_m1 = 0
        psi_N = 0
        phi_m1 = 0
        p_m1 = 0
        p_N = 0
        V_m1 = 0
        V_N = 0
        
        Psi=np.zeros(N+2,dtype=complex)
        Psi[0]=psi_m1
        Psi[1:N+1]=SOL[0:N]
        Psi[N+1]=psi_N
        
        Phi=np.zeros(N+2,dtype=complex)
        Phi[0]=phi_m1
        Phi[1:N+2]=SOL[N:2*N+1]
        
        P=np.zeros(N+2,dtype=complex)
        P[0]=p_m1
        P[1:N+1]=SOL[2*N+1:3*N+1]
        P[N+1]=p_N
        
        V=np.zeros(N+2,dtype=complex)
        V[0]=V_m1
        V[1:N+1]=SOL[3*N+1:]
        V[N+1]=V_N
        
        return Psi,Phi,P,V

    def _Problem_Solution_General_Case(self,omega):
        """
        General case
        """
        
        if self.eta!=0:
            Psi,Phi,P,V = self._Problem_Solution_Case_Psi_eq_1(omega)
        else:
            if self.nu==0:
                Psi,Phi,P,V = self._Problem_Solution_Case_Psi_eq_2(omega)
            else:
                Psi,Phi,P,V = self._Problem_Solution_Case_Phi_eq_4(omega)

        return Psi,Phi,P,V



    def _Field_N(self,omega):
        """
        Returns the value Field_N of Field in r=1, where Field is Psi or Phi, depending on the choice of the parameters
        In the cases where eta=0, or nu=0, or chiperp=0, different routines have to be used.
        """

        if self.eta!=0:
            Psi,Phi,P,V = self._Problem_Solution_Case_Psi_eq_1(omega)
            val_N = Psi[-1]
        else:
            if self.nu==0:
                Psi,Phi,P,V = self._Problem_Solution_Case_Psi_eq_2(omega)
                val_N = Psi[-1]
            else:
                Psi,Phi,P,V = self._Problem_Solution_Case_Phi_eq_4(omega)
                val_N = Phi[-1]

        return val_N

    def _Field_N_real(self,gamma):
        """
        Returns the real value Psi_N of Psi in r=1 for wstar=0.
        """
        
        if self.di!=0:
            print " "
            print " _Field_N_real: Warning: The routine _Field_N_real should be used with di=0 only"
            
        val_N = self._Field_N(1j*gamma)

        return val_N.real

    def _Solve_For_Gamma(self,gamma_max,gamma_min):
        """
        Define problem
        Solve using brent
        """

        if self.di!=0:
            print " "
            print " Solve_For_Gamma: Warning: The routine Solve_For_Gamma should be used with di=0 only"

        def F(gamma):
            return self._Field_N_real(gamma)

        try:
            root,out=brentq(F,gamma_max,gamma_min,full_output=True,xtol=1.e-15)
            if self.eta!=0:
                Psi,Phi,P,V = self._Problem_Solution_Case_Psi_eq_1(1j*root)
            else:
                if self.nu==0:
                    Psi,Phi,P,V = self._Problem_Solution_Case_Psi_eq_2(1j*root)
                else:
                    Psi,Phi,P,V,omega_sol,out_dum = self._Solve_For_Omega(root*1j)
                    root = omega_sol.imag

        except ValueError:
            return None,None,None,None,None,None

        return Psi,Phi,P,V,root,out 

    def _Solve_For_Omega(self,omega_guess,thresh_npts=20):
        """
        Solve for the complex frequency using the scipy.optimize function root
        The function works in R^2 instead of C, hence we first transform C to R^2
        """

        def F(omega_R2):
            if np.isnan(omega_R2[0]):
                return None
            else:
                omega_comp = omega_R2[0] + 1j*omega_R2[1]
                sys.stdout.write(" omega = "+'({:1.4e},{:1.4e})'.format(omega_comp.real,omega_comp.imag)+'\r')
                sys.stdout.flush()
                res = self._Field_N(omega_comp)
                if np.abs(omega_comp)/np.abs(omega_guess)>2:
                    return None
                else:
                    return [res.real,res.imag]

        omega_R2_guess = [omega_guess.real,omega_guess.imag]

        try:
            sol=root(F,omega_R2_guess,method='hybr')
            success=sol.success
            message=sol.message
        
        except RuntimeError:
            success=False
            message='Could not converge toward a solution, try with an other guess'
    
        if success:
            omega_sol = sol.x[0]+sol.x[1]*1j
    
            Psi,Phi,P,V=self._Problem_Solution_General_Case(omega_sol)

            is_OK = self._Is_First_Harmonic(Psi,Phi,tol=1.e-6)

            if is_OK:
                # try:
                #     iota,pres,p_r,Omega_r,epsilon,x,X,dx = profiles(profs,N)
                #     nmi_fun = lambda x: self.m*self.iota(x) -self.n
                #     kpar = nmi_fun(self.X)
                #     i_rs = self._cont_2_ind(0,kpar)
                #     rs = self.X[i_rs]
                #     delta = self._Sheet_Width_Phi(self.X,Phi,x0=rs)
                #     npts = round(delta/self.dx)
                #     if npts<thresh_npts:
                #         print " "
                #         print " The problem is under-resolved"
                #         print " There are less than {:d} points such that Phi > FWHM(Phi)".format(thresh_npts)
                #         print " You should increase the resolution"
                #         print " "
                # except ValueError:
                #     print " "
                #     print " Failed measuring the Sheet Width"

                # print " "+message
                return Psi,Phi,P,V,omega_sol,sol
            else:
                print " "
                print " The solution does not seem to be an even mode."
                print " I return the solution anyway but you are responsible for what you are doing."
                print " "
                return Psi,Phi,P,V,omega_sol,sol
        else:        
            print " "
            print " "+message
            print " "
            return None,None,None,None,None,None

    def _Scan_Field_N(self,gamma_max,gamma_min,ntestpoints=100,plot=False,fignumber=1):
        """
        Perform a log scan from gamma_max to gamma_min, with di = 0
        If plot, plot the result
        """

        if self.di!=0:
            print " "
            print " Scan_Field_N: Warning: The routine Scan_Field_N should be used with di=0 only"

        gammas=np.logspace(np.log10(gamma_max),np.log10(gamma_min),ntestpoints)
        vals_N=np.zeros(ntestpoints)

        i=0
        for gamma in gammas:
            sys.stdout.write(" gamma = "+str("%1.6e" % gamma)+'\r')
            sys.stdout.flush()
            vals_N[i]=self._Field_N_real(gamma)
            i=i+1
    
        if plot:
            plt.figure(fignumber)
            plt.clf()
            plt.loglog(gammas,np.abs(vals_N),'sg-',lw=2,ms=3)
            plt.grid(True)
            plt.figure(fignumber+1)
            plt.clf()
            plt.semilogx(gammas,vals_N,'sg-',lw=2,ms=3)
            plt.grid(True)
            
        return gammas,vals_N

    def GrowthRateSearch(self,gamma_max=None,gamma_min=None,ntestpoints=100,i=0):
        """
        Gives the Growth rate of the first harmonic in the Problem
        You must give bounds for the growth rate search, gamma_max and gamma_min
        """

        if not self.defined:
            print " "
            print " You must first define the pressure and iota profiles with DefinePressureIotaProfiles !"
            return

        if gamma_max==None:
            gamma_max = raw_input(" gamma_max (default 1e-1) = ")
        if gamma_max=='':
            gamma_max = 1e-1
        else:
            gamma_max = float(gamma_max)

        if gamma_min==None:
            gamma_min = raw_input(" gamma_min (default 1e-2) = ")
        if gamma_min=='':
            gamma_min = 1e-2
        else:
            gamma_min = float(gamma_min)

        i=i+1
        gammas,vals_N = self._Scan_Field_N(gamma_max,gamma_min,ntestpoints=ntestpoints)

        fignumber = 100+i

        plt.figure(fignumber)
        plt.clf()
        plt.semilogx(gammas,vals_N,'sg-')
        plt.grid(True)
        plt.xlim([np.min(gammas),np.max(gammas)])

        # First ask what to do
        instruction=self._WWYLTD()
        instruction,outarg = self._Perform_Task(instruction,fignumber)

        # if (instruction=='z') or (instruction=='g') or (instruction=='i') or (instruction=='s'):
        if (instruction=='z') or (instruction=='s'):
            gamma_min = outarg[0]
            gamma_max = outarg[1]
        # if (instruction=='z') or (instruction=='i'):
        if (instruction=='z'):
            # Recursive call in the new interval
            self.GrowthRateSearch(gamma_max,gamma_min,ntestpoints=ntestpoints,i=i)

        # if (instruction=='s') or (instruction=='g'):
        if (instruction=='s'):
            Psi,Phi,P,V,gamma_sol,out=self._Solve_For_Gamma(gamma_max,gamma_min)

            is_OK = self._Is_First_Harmonic(Psi,Phi)
            is_not_disc = self._Is_Not_On_Discontinuity(Psi)
    
            if is_OK:
                print " The solution "+out.flag
                print str(" Solution is gamma = %1.5e" % gamma_sol)
                print str(" Number of iterations %d" % out.iterations)

                self.omega = gamma_sol*1j
                self.solution = self._Solution(gamma_sol*1j,Psi,Phi,P,V,out)
                self.solutions.append(self.solution)
                self.HasOneSolution = True

            elif (not is_OK) and is_not_disc:
                print str(" At this stage gamma is = %1.5e" % gamma_sol)
                print str(" This may not be the most unstable mode, maybe try changing the parameters gamma_max and gamma_min")

                self.omega = gamma_sol*1j
                self.solution = self._Solution(gamma_sol*1j,Psi,Phi,P,V,out)
                self.solutions.append(self.solution)
                self.HasOneSolution = True

            else:
                print str(" At this stage gamma is = %1.5e" % gamma_sol)
                print str(" We ended up on the discontinuity, trying again")
                print str(" Not returning any solution, try again")

        elif instruction=='q':
            # Nothing to do
            return
            
        # Return the Solution as a dictionary
            
                         
    def _WWYLTD(self):
        """
        The name of this function means What Would You Like To Do ?
        
        It returns instruction (= What I would like to do :-) )
        
        The most common use will be the following :
        - First zoom on the figure in order to be centered
        on the discontinuity (the solution is always close to a discontinuity)
        Then type 'z' then 'enter' (or simply 'enter' since it is the default) and
        you will have to provide two horizontal positions by clicking on the figure
        The program will then increase the resolution in this interval and plot a new figure
        - If the resolution of the new figure is sufficient so that you can see the curve
        going through the zero line, just zoom around the zero (avoiding the discontinuity)
        and finally simply press 's' and then 'enter' in order to solve with the Newton
        """

        # print " What would you like to do ? (z,g,i,q,s)(default z)"
        print " What would you like to do ? (z,q,s)(default z)"
        print " "
        print " z = zoom on the figure and perform a new search (default)"
        # print " g = select 2 points by ginput for solving"
        # print " i = input other search region"
        print " q = quit"
        print " s = solve with the current provided interval (make sure it is appropriate)"
        print " "
        
        instruction = raw_input(" Your choice: ")
        if instruction == '':
            instruction = 'z'

        return instruction

    def _Perform_Task(self,instruction,fignumber):
        """
        Takes a string of instruction (see WWYLTD)
        Performs the corresponding instruction if there is something to do
        """
        
        if instruction=='z':
            plt.figure(fignumber)
            Selection = plt.ginput(2,timeout=3600)
            outarg = [Selection[0][0], Selection[1][0]]
        # elif instruction=='g':
        #     plt.figure(fignumber)
        #     Selection = plt.ginput(2,timeout=3600)
        #     x = [Selection[0][0], Selection[1][0]]
        #     outarg = x
        # elif instruction=='i':
        #     gamma = raw_input(" Input the desired interval as gamma0,gamma1: ")
        #     gamma0 = float(gamma.split(',')[0])
        #     gamma1 = float(gamma.split(',')[1])
        #     outarg = [gamma0,gamma1]
        elif instruction=='s':
            plt.figure(fignumber)
            gamma0 = plt.xlim()[0]
            gamma1 = plt.xlim()[1]
            outarg = [gamma0,gamma1]
        else:
            outarg = None

        return instruction,outarg

    def _Is_First_Harmonic(self,Psi,Phi,tol=1.e-6):
        """
        Takes Psi and Phi solutions and checks the following
        Phi does not change sign
        Psi(r=1) = 0 (up to tol)
        
        If OK, return True
        """

        if np.abs(Psi[-1])/np.max(np.abs(Psi))<tol:
            Psi_OK=True
        else:
            Psi_OK=False
    
        # Find change of signs in Phi[0:-1]
        if np.max(Phi.imag)<1e-10:
            # This is the easy case
            Phi=Phi.real
            liste=(np.sign(Phi[0:-2])*np.sign(Phi[1:-1]))
            indices = [i for i, ind in enumerate(liste) if int(ind) == -1]

            if len(indices)==0:
                Phi_OK = True
            else:
                Phi_OK = False
        elif np.max(Phi.real)<1e-10:
            # This is the easy case
            Phi=Phi.imag
            liste=(np.sign(Phi[0:-2])*np.sign(Phi[1:-1]))
            indices = [i for i, ind in enumerate(liste) if int(ind) == -1]
            
            if len(indices)==0:
                Phi_OK = True
            else:
                Phi_OK = False
        else:
            # We cannot use the modulus of Phi since it is always positive.
            # Conjecture : We should see phase jumps... Let's check that next time it shows up
            phase=np.angle(Phi)
            
            dp = np.diff(phase[1:-1])
            
            if np.max(np.abs(dp))>0.5*np.pi:
                Phi_OK = False
            else:
                Phi_OK = True

        print " "
        if Phi_OK and Psi_OK:
            return True
        else:
            print str(" This is not a first harmonic, try changing the parameters gamma_max and gamma_min")
            print " Phi_OK  = "+str(Phi_OK)
            print " Psi_OK = "+str(Psi_OK)
            if not Psi_OK:
                res = np.abs(Psi[-1])/np.max(np.abs(Psi))
                print str(" np.abs(Psi[-1])/np.max(np.abs(Psi)) = %1.2e" % res)
            return False

    def _Is_Not_On_Discontinuity(self,Psi,thresh=1e2):
        """
        If the max of Psi is more than thresh (by experience, 1e2 sould be largely sufficient), then
        it means the algorithm was mistaken and somehow ended up on the discontinuity.
        In this case, we want to discard the solution
        """

        is_OK = True
        
        if np.max(np.abs(Psi))>thresh:
            is_OK = False
            
        return is_OK


    def _MHD_Displacement(self,Phi):
        """
        Gives the shape of the MHD displacement corresponding to Phi
        We don't care about any factors in fron so this is simply Phi/X
        """
        
        Xi = np.zeros(len(Phi),dtype=complex)
        X = np.linspace(0,1,len(Phi))
        Xi[1:] = Phi[1:]/X[1:]
        Xi[0] = Xi[1]
        
        return Xi

    def _Current(self,Psi,m):
        """
        Gives the current as nabla_perp^2(Psi)
        """

        J = np.zeros(len(Psi),dtype=complex)
        X = np.linspace(0,1,len(Psi))
        dx1 = 1./(X[1]-X[0])
        
        J[1:-1] = ( (Psi[0:-2] + Psi[2:] - 2*Psi[1:-1])*dx1**2
                    +0.5*(Psi[2:] - Psi[0:-2])*dx1/X[1:-1]
                    - m**2*Psi[1:-1]/X[1:-1]**2 )
        
        return J

    def GetSolution(self,solution=None):
        """
        Simply return the solution as arrays
        """

        # if solution==None:
        #     # In this case read the standard solution
        #     X = self.X
        #     Psi = self.solution['Psi']
        #     J = self._Current(Psi,self.m)
        #     Phi = self.solution['Phi']
        #     Xi = self._MHD_Displacement(Phi)
        #     P = self.solution['P']
        #     V = self.solution['V']
        #     omega = self.solution['omega']
        # else:
        if solution==None:
            # In this case use the last computed solution
            solution = self.solutions[-1]

        # Otherwise read what is in solution
        X = np.linspace(0,1,solution['N']+2)
        Psi = solution['Psi']
        J = self._Current(Psi,solution['m'])
        Phi = solution['Phi']
        Xi = self._MHD_Displacement(Phi)
        P = solution['P']
        V = solution['V']
        omega = solution['omega']

        return X,Psi,J,Phi,Xi,P,V,omega

    def PrintParameters(self,solution=None):
        """
        Prints the physical parameters in the dictionary solution
        """

        # if solution==None:
        #     beta = self.beta
        #     eta = self.eta
        #     nu = self.nu
        #     chiperp = self.chiperp
        #     chipar = self.chipar
        #     # compr = self.compr
        #     di = self.di
        #     tau = self.tau
        #     m = self.m
        #     n = self.n
        #     N = self.N
        #     epsilon = self.epsilon
        # else:

        if solution==None:
            try:
                solution = self.solutions[-1]
                beta = solution['beta']
                eta = solution['eta']
                nu = solution['nu']
                chiperp = solution['chiperp']
                chipar = solution['chipar']
                # compr = solution['compr']
                di = solution['di']
                tau = solution['tau']
                m = solution['m']
                n = solution['n']
                N = solution['N']
                epsilon = solution['epsilon']
            except IndexError:
                # It means we have computed nothing yet
                # In this case simply print the current parameters
                beta = self.beta
                eta = self.eta
                nu = self.nu
                chiperp = self.chiperp
                chipar = self.chipar
                # compr = self.compr
                di = self.di
                tau = self.tau
                m = self.m
                n = self.n
                N = self.N
                epsilon = self.epsilon
        else:
            beta = solution['beta']
            eta = solution['eta']
            nu = solution['nu']
            chiperp = solution['chiperp']
            chipar = solution['chipar']
            # compr = solution['compr']
            di = solution['di']
            tau = solution['tau']
            m = solution['m']
            n = solution['n']
            N = solution['N']
            epsilon = solution['epsilon']

        print '-----------------------------------------------------------------------------------------------------------------------------------------------'
        print '|     beta    |     eta     |     nu      |   chiperp   |   chipar    |     di      |     tau     |    m     |    n     |   N   |   epsilon   |'
        print '-----------------------------------------------------------------------------------------------------------------------------------------------'
        print '| {:1.5e} | {:1.5e} | {:1.5e} | {:1.5e} | {:1.5e} | {:1.5e} | {:1.5e} | {:1.2e} | {:1.2e} | {:5d} | {:1.5e} |'.format(beta,eta,nu,chiperp,chipar,di,tau,m,n,N,epsilon)
        print '-----------------------------------------------------------------------------------------------------------------------------------------------'

    def PlotSolution(self,fignumber=1,solution=None):
        """
        Plot a figure containing the normalized eigenfunctions
        """
    
        X,Psi,J,Phi,Xi,P,V,omega = self.GetSolution(solution)

        rs = self.x[abs(self.nmi).argmin()]
        
        plt.figure(fignumber,figsize=(20,13))
        plt.clf()
        gs = matplotlib.gridspec.GridSpec(2,3)
        gs.update(left=0.07, right=0.97,bottom=0.08,hspace=0.25,wspace=0.3,top=0.95)

        # First subfigure: Phi
        ax=plt.subplot(gs[0])
        plt.xlabel('Radius')
        plt.ylabel('Potential')
        ax.title.set_text(r'$\Phi$')
        phimax = abs(Phi).max()
        phimin = abs(Phi).min()
        phimax = max(abs(phimax),abs(phimin))
        plt.plot(X,Phi.real/phimax,'g-',lw=2,label='Real part')
        plt.plot(X,Phi.imag/phimax,'r-',lw=2,label='Imaginary part')
        plt.legend(loc=2,prop={'size':12})
        plt.grid(True)
        ylim=ax.get_ylim()
        plt.plot([rs,rs],ylim,'k--')
        plt.axis([0,1,ylim[0],ylim[1]])
        self._set_power_limits()

        # Second subfigure: Xi
        ax=plt.subplot(gs[3])
        plt.xlabel('Radius')
        plt.ylabel('MHD Displacement')
        ax.title.set_text(r'$\xi$')
        ximax = abs(Xi).max()
        ximin = abs(Xi).min()
        ximax = max(abs(ximax),abs(ximin))
        plt.plot(X,Xi.real/ximax,'g-',lw=2,label='Real part')
        plt.plot(X,Xi.imag/ximax,'r-',lw=2,label='Imaginary part')
        plt.legend(loc=2,prop={'size':12})
        plt.grid(True)
        ylim=ax.get_ylim()
        plt.plot([rs,rs],ylim,'k--')
        plt.axis([0,1,ylim[0],ylim[1]])
        self._set_power_limits()

        # Third subfigure: Psi
        ax=plt.subplot(gs[1])
        plt.xlabel('Radius')
        plt.ylabel('Magnetic Flux')
        ax.title.set_text(r'$\psi$')
        psimax = abs(Psi).max()
        psimin = abs(Psi).min()
        psimax = max(abs(psimax),abs(psimin))
        plt.plot(X,Psi.real/psimax,'g-',lw=2,label='Real part')
        plt.plot(X,Psi.imag/psimax,'r-',lw=2,label='Imaginary part')
        plt.legend(loc=2,prop={'size':12})
        plt.grid(True)
        ylim=ax.get_ylim()
        plt.plot([rs,rs],ylim,'k--')
        plt.axis([0,1,ylim[0],ylim[1]])
        self._set_power_limits()

        # Fourth subfigure: J
        ax=plt.subplot(gs[4])
        plt.xlabel('Radius')
        plt.ylabel('Current')
        ax.title.set_text(r'$J$')
        jmax = abs(J).max()
        jmin = abs(J).min()
        jmax = max(abs(jmax),abs(jmin))
        plt.plot(X,J.real/jmax,'g-',lw=2,label='Real part')
        plt.plot(X,J.imag/jmax,'r-',lw=2,label='Imaginary part')
        plt.legend(loc=2,prop={'size':12})
        plt.grid(True)
        ylim=ax.get_ylim()
        plt.plot([rs,rs],ylim,'k--')
        plt.axis([0,1,ylim[0],ylim[1]])
        self._set_power_limits()

        # Fifth subfigure: P
        ax=plt.subplot(gs[2])
        plt.xlabel('Radius')
        plt.ylabel('Pressure perturbation')
        ax.title.set_text(r'$P$')
        pmax = abs(P).max()
        pmin = abs(P).min()
        pmax = max(abs(pmax),abs(pmin))
        plt.plot(X,P.real/pmax,'g-',lw=2,label='Real part')
        plt.plot(X,P.imag/pmax,'r-',lw=2,label='Imaginary part')
        plt.legend(loc=3,prop={'size':12})
        plt.grid(True)
        ylim=ax.get_ylim()
        plt.plot([rs,rs],ylim,'k--')
        plt.axis([0,1,ylim[0],ylim[1]])
        self._set_power_limits()

        # Sixth subfigure: V
        ax=plt.subplot(gs[5])
        plt.xlabel('Radius')
        plt.ylabel('Vorticity')
        ax.title.set_text(r'$\nabla^2_\perp\Phi$')
        vmax = abs(V).max()
        vmin = abs(V).min()
        vmax = max(abs(vmax),abs(vmin))
        plt.plot(X,V.real/vmax,'g-',lw=2,label='Real part')
        plt.plot(X,V.imag/vmax,'r-',lw=2,label='Imaginary part')
        plt.legend(loc=2,prop={'size':12})
        plt.grid(True)
        ylim=ax.get_ylim()
        plt.plot([rs,rs],ylim,'k--')
        plt.axis([0,1,ylim[0],ylim[1]])
        self._set_power_limits()

    def _Extrapolate_Comp(self,X,Y,x):
        """
        Takes a series of (x,y) couples and constructs a 2nd order polynomial using the last values of the arrays in order to extrapolate to the next x value
        
        Exception when the array contains only 1 or 2 values
        
        When 2 linear extrapolation
        When 1, equal value is returned
        """
        
        n=len(X)
        
        Yr = Y.real
        Yi = Y.imag
   
        if n==1:
            # gamma_I = Y[0].imag
            # freq = 0.5*x
            # gamma = gamma_I*np.sqrt(1-(0.5*x/gamma_I)**2)
            # y=freq+1j*gamma
            y = Y[0]
        elif n==2:
            # Linear extrapolation
            y = Y[1] + (x-X[1])/(X[1]-X[0])*(Y[1]-Y[0])
        else:
            # Order 2 extrapolation using last 3 points
            pr = np.polyfit(X[-3:],Yr[-3:],2)
            yr = np.polyval(pr,x)
            pi = np.polyfit(X[-3:],Yi[-3:],2)
            yi = np.polyval(pi,x)
            y = yr + 1j*yi
            
        return y

    def _nmi_fun(self,x):
        """
        nmi = m*iota-n
        """
        
        return self.m*self.iota(x) - self.n

    def ParameterScan(self,parameter_name,parameter_maxmin_val,gamma_maxmin_val=None,adaptative_resolution=True,maxcoeff=0.1,nfmax=20,nfmin=8,dparini=1e-8,dparini_coeff=1e-2,thresh_npts=40):
        """
        Here we make a parameter study where the parameter is varied dynamically
        
        The varied parameter is determined by the input which has more than one value
        We allow numpy arrays and lists
        """

        if not self.HasOneSolution:
            print ' You must have obtained at least one solution !'
            print ' First obtain one solution with GrowthRateSearch method'
            return

        parameter_count = 0
        
        solutions = []

        if parameter_name=='beta':
            params = [self.solutions[-1]['beta']]
        elif parameter_name=='eta':
            params = [self.solutions[-1]['eta']]
        elif parameter_name=='nu':
            params = [self.solutions[-1]['nu']]
        elif parameter_name=='chiperp':
            params = [self.solutions[-1]['chiperp']]
        elif parameter_name=='chipar':
            params = [self.solutions[-1]['chipar']]
        elif parameter_name=='compr':
            params = [self.solutions[-1]['compr']]
        elif parameter_name=='di':
            params = [self.solutions[-1]['di']]
        elif parameter_name=='tau':
            params = [self.solutions[-1]['tau']]
        elif parameter_name=='m_and_n':
            params = [self.solutions[-1]['m']]
            nsurm = self.solutions[-1]['n']/self.solutions[-1]['m']
        elif parameter_name=='m_only':
            params = [self.solutions[-1]['m']]
        elif parameter_name=='n_only':
            params = [self.solutions[-1]['n']]
        elif parameter_name=='epsilon':
            params = [self.solutions[-1]['epsilon']]

        if parameter_maxmin_val>params[0]:
            increasing=True
        else:
            increasing=False

        # First value of the parameter
        param = params[0]
        if increasing:
            if param==0:
                dpar = dparini
            else:
                dpar = dparini_coeff*param
        else:
            if param==0:
                dpar = -dparini
            else:
                dpar = -dparini_coeff*param

        omegas_par = [self.solutions[-1]['omega']]
        
        cond = True
        while cond:
            success = False
            i=0
            while not success:
                i=i+1
                if increasing:
                    param = min(params[-1] + dpar,parameter_maxmin_val)
                else:
                    param = max(params[-1] + dpar,parameter_maxmin_val)
                if i==1:
                    print ' ---------------------------------------------------------------------------------------------------------------'
                print ' Trying with {:} = {:1.5e} :'.format(parameter_name,param)
                omegas_comp = np.array(omegas_par,dtype=complex)
                omega_guess = self._Extrapolate_Comp(params,omegas_comp,param)
                if self.di==0:
                    omega_guess = omega_guess.imag*1j

                if parameter_name=='beta':
                    self._ResetParameters(beta=param)
                elif parameter_name=='eta':
                    self._ResetParameters(eta=param)
                elif parameter_name=='nu':
                    self._ResetParameters(nu=param)
                elif parameter_name=='chiperp':
                    self._ResetParameters(chiperp=param)
                elif parameter_name=='chipar':
                    self._ResetParameters(chipar=param)
                elif parameter_name=='compr':
                    self._ResetParameters(compr=param)
                elif parameter_name=='di':
                    self._ResetParameters(di=param)
                elif parameter_name=='tau':
                    self._ResetParameters(tau=param)
                elif parameter_name=='m_and_n':
                    self._ResetParameters(m=param,n=param*nsurm)
                elif parameter_name=='m_only':
                    self._ResetParameters(m=param)
                elif parameter_name=='n_only':
                    self._ResetParameters(n=param)
                elif parameter_name=='epsilon':
                    self._ResetParameters(epsilon=param)
                # Try to obtain the new solution with the new value of the parameter
                Psi,Phi,P,V,omega,sol=self._Solve_For_Omega(omega_guess)
                if sol==None:
                    print ' No solution found, reducing step'
                    print ' Changed {:} step to d{:} = {:1.2e}'.format(parameter_name,parameter_name,dpar)
                    dpar = dpar*0.1
                else:
                    success = True
                    parameter_count = parameter_count+1
                    if sol.nfev > nfmax:
                        dpar = dpar*2./3.
                        print ' Too many iterations, reducing {:} step'.format(parameter_name)
                        print ' Changed {:} step to d{:} = {:1.2e}'.format(parameter_name,parameter_name,dpar)
                    elif sol.nfev < nfmin:
                        # Increase step but prevent too large steps, here we want at least around 10 steps by decade
                        if increasing:
                            dpar = min(abs(dpar*4./3.),maxcoeff*param)
                        else:
                            dpar = -min(abs(dpar*4./3.),maxcoeff*param)
                        print ' Low number of iterations, I increase the {:} step'.format(parameter_name)
                        print ' Changed {:} step to d{:} = {:1.2e} :'.format(parameter_name,parameter_name,dpar)
            # Check resolution and increase N if resolution is too small
            try:
                #nmi_fun = lambda x: self.m*self.iota(x) - self.n
                rs = self.X[abs(self._nmi_fun(self.X)).argmin()]
                delta = self._Sheet_Width_Phi(Phi,rs)
                dx = self.X[1]-self.X[0]
                npts = round(delta/dx)
                if (npts < thresh_npts) and adaptative_resolution:
                    print " "
                    print " Increasing the resolution by a factor of 2"
                    print " "
                    self.N = (self.N+1)*2-1
                    self.X,self.dx = np.linspace(0,1,self.N+2,retstep=True)
                    self.x = self.X[1:self.N+2]
                    self._ResetParameters()
            except ValueError:
                print " "
                print " Failed measuring the Sheet Width"

            omegas_par.append(omega)
            self.omega = omega
            params.append(param)
            solution = self._Solution(omega,Psi,Phi,P,V,sol)
            solutions.append(solution)

            if increasing:
                cond=params[-1]<parameter_maxmin_val 
            else:
                cond=params[-1]>parameter_maxmin_val 

            if gamma_maxmin_val!=None:
            # Stop the calculation if gamma is larger or smaller than gamma_maxmin_val 
                if gamma_maxmin_val>gamma0:
                    # Increasing case
                    if self.omega.imag>gamma_maxmin_val:
                        break
                elif gamma_maxmin_val<gamma0:
                    # Decreasing case
                    if self.omega.imag<gamma_maxmin_val:
                        break

        if self._parameter_names!=[]:
            if parameter_name==self._parameter_names[-1]:
                print "\n ========================================================="
                print " || Fusioning this parameter scan with the previous one ||"
                print " ========================================================="
                self._parameter_counts[-1] = self._parameter_counts[-1] + len(solutions[1:])
                # append all solutions except the first one
                for solution in solutions[1:]:
                    self.solutions.append(solution)            
            else:
                self._parameter_names.append(parameter_name)
                self._parameter_counts.append(parameter_count)
                for solution in solutions:
                    self.solutions.append(solution)
        else:
            self._parameter_names.append(parameter_name)
            self._parameter_counts.append(parameter_count)
            for solution in solutions:
                self.solutions.append(solution)

    def PrintScanHistory(self):
        """
        Print the history of the past scans, based on the contents of self.solutions
        """

        nscan = len(self._parameter_counts)

        # Summary
        print ' '
        print ' Total number of scans: {:d}'.format(nscan)
        print ' '

        print ' Initial parameters'
        self.PrintParameters(solution=self.solutions[0])
        print ' '

        ini = 0
        end = self._parameter_counts[0]
        for i in range(nscan):
            mnmode=None
            parname = self._parameter_names[i]
            if parname=='m_only':
                parname = 'm'
                mnmode='m_only'
            elif parname=='n_only':
                parname = 'n'
                mnmode='n_only'
            elif parname=='m_and_n':
                parname = 'm'
                mnmode='m_and_n'
            inival = self.solutions[ini][parname]
            endval = self.solutions[end][parname]
            if mnmode=='m_and_n':
                print ' Scan No{:2d}: m and n were varied together from {:1.3e} to {:1.3e}. Scan contained in solutions[{:d}] to solutions[{:d}]'.format(i+1,inival,endval,ini,end) 
            else:
                print ' Scan No{:2d}: {:} was varied from {:1.3e} to {:1.3e}. Scan contained in solutions[{:d}] to solutions[{:d}]'.format(i+1,parname,inival,endval,ini,end) 
            if i!=nscan-1:
                ini = end
                end = end + self._parameter_counts[i+1]



    def _Solution(self,omega,Psi,Phi,P,V,sol):
        """
        Put all the parameters, eigenvalues and eigenfunctions in a dictionary
        """

        return {"beta" : self.beta,
                "eta" : self.eta,
                "nu" : self.nu,
                "chiperp" : self.chiperp,
                "chipar" : self.chipar,
                "compr" : self.compr,
                "di" : self.di,
                "tau" : self.tau,
                "m" : self.m,
                "n" : self.n,
                "N" : self.N,
                "epsilon" : self.epsilon,
                "omega" : omega,
                "Psi" : Psi,
                "Phi" : Phi,
                "P" : P,
                "V" : V,
                "sol" : sol}


    def _Sheet_Width_Phi(self,Phi,x0,from_max=False):
        """
        Determine the FWHM of Phi
        """
            
        # Position of resonance
        if from_max:
            i0 = np.abs(Phi).argmax()
        else:
            i0 = self._cont_2_ind(self.X,x0)

        i_right = i0+self._cont_2_ind(Phi[i0:],np.max(Phi)*0.5)
        i_left = self._cont_2_ind(Phi[:i0],np.max(Phi)*0.5)
    
        delta = self.X[i_right] - self.X[i_left]

        return delta


    def Rewind(self):
        """
        Reset the parameters to what they were before the last parameter study
        If erase==True (default), also remove the last data
        """

        if not self._parameter_counts==[]:
            self.solutions = self.solutions[:-self._parameter_counts[-1]]
            self._parameter_counts=self._parameter_counts[:-1]
            self._parameter_names=self._parameter_names[:-1]
            self._ResetParameters(beta=self.solutions[-1]['beta'],
                                 eta=self.solutions[-1]['eta'],
                                 nu=self.solutions[-1]['nu'],
                                 chiperp=self.solutions[-1]['chiperp'],
                                 chipar=self.solutions[-1]['chipar'],
                                 compr=self.solutions[-1]['compr'],
                                 di=self.solutions[-1]['di'],
                                 tau=self.solutions[-1]['tau'],
                                 m=self.solutions[-1]['m'],
                                 n=self.solutions[-1]['n'],
                                 N=self.solutions[-1]['N'],
                                 epsilon=self.solutions[-1]['epsilon'])
            self.omega = self.solutions[-1]['omega']
                                 
    def PlotGrowthRatesAndFrequencies(self,study_number,fignumber=1,log=None):
        """
        plots the growth rates and frequencies for the last parameter study
        """

        if self._parameter_counts==[]:
            print ' You must first perform a parameter study'
            return

        params,omegas = self.GetGrowthRatesAndFrequencies(study_number)

        parameter_name = self._parameter_names[study_number-1]

        if (parameter_name=='beta' or 
            parameter_name=='eta' or 
            parameter_name=='nu' or 
            parameter_name=='tau' or
            parameter_name=='epsilon'):
            string = '\\' + parameter_name
        elif parameter_name=='chiperp':
            string = '\chi_\perp'
        elif parameter_name=='chipar':
            string = '\chi_\|'
        elif parameter_name=='compr':
            string = '\Gamma'
        elif parameter_name=='di':
            string = 'd_i'
        elif parameter_name=='m':
            string = 'm'
        elif parameter_name=='n':
            string = 'n'
        elif parameter_name=='N':
            string = 'N'
            
        plt.figure(fignumber,figsize=(8,13))
        plt.clf()
        gs = matplotlib.gridspec.GridSpec(2,1)
        gs.update(left=0.15, right=0.91,bottom=0.08,hspace=0.25,wspace=0.3,top=0.95)

        # First Subfigure : Growth rate
        ax=plt.subplot(gs[0])
        plt.xlabel(r'${:}$'.format(string))
        plt.ylabel(r'Growth rate $\gamma$')
        if log==None:
            plt.plot(params,omegas.imag,'g-',lw=2)
            self._set_power_limits()
        elif log=='semilogx':
            plt.semilogx(params,omegas.imag,'g-',lw=2)
        elif log=='semilogy':
            plt.semilogy(params,omegas.imag,'g-',lw=2)
        elif log=='loglog':
            plt.loglog(params,omegas.imag,'g-',lw=2)
        plt.grid(True)
        ax.title.set_text('Growh rate')

        # Second Subfigure : Frequency
        ax=plt.subplot(gs[1])
        plt.xlabel(r'${:}$'.format(string))
        plt.ylabel(r'Frequency $\omega_r$')
        if log==None:
            plt.plot(params,omegas.real,'r-',lw=2)
            self._set_power_limits()
        elif log=='semilogx' or log=='semilogy' or log=='loglog':
            # For the frequency it can be positive or negative so semilogy and loglog not allowed
            plt.semilogx(params,omegas.real,'r-',lw=2)
        plt.grid(True)
        ax.title.set_text('Frequency')

    def GetGrowthRatesAndFrequencies(self,study_number):
        """
        gets the growth rates and frequencies for the parameter study study_number (starts at 1, not 0 !)
        """
        
        if self._parameter_counts==[]:
            print ' You must first perform a parameter study'
            return

        params = np.zeros(self._parameter_counts[study_number-1]+1)
        omegas = np.zeros(self._parameter_counts[study_number-1]+1,dtype=complex)

        if study_number==1:
            offset=0
        else:
            offset=0
            for j in range(study_number-1):
                offset=offset + self._parameter_counts[j]

        
        for i in range(self._parameter_counts[study_number-1]+1):
            index = offset + i
            solution = self.solutions[index]
            params[i] = solution[self._parameter_names[study_number-1]]
            omegas[i] = solution['omega']

        return params,omegas

    def Save(self,filename=None):
        """
        Save all the fields so far in a file
        Create the filename automatically, based on first and last parameters, if no filename is provided
        """

        if filename==None:
            sol1 = self.solutions[0]
            sol2 = self.solutions[-1]

            if sol1['beta']==sol2['beta']:
                filename = 'beta_{:1.1e}'.format(sol1['beta'])
            else:
                filename = 'beta_from_{:1.1e}_to_{:1.1e}'.format(sol1['beta'],sol2['beta'])

            if sol1['eta']==sol2['eta']:
                filename = filename+'_eta_{:1.1e}'.format(sol1['eta'])
            else:
                filename = filename+'_eta_from_{:1.1e}_to_{:1.1e}'.format(sol1['eta'],sol2['eta'])

            if sol1['nu']==sol2['nu']:
                filename = filename+'_nu_{:1.1e}'.format(sol1['nu'])
            else:
                filename = filename+'_nu_from_{:1.1e}_to_{:1.1e}'.format(sol1['nu'],sol2['nu'])

            if sol1['chiperp']==sol2['chiperp']:
                filename = filename+'_chiperp_{:1.1e}'.format(sol1['chiperp'])
            else:
                filename = filename+'_chiperp_from_{:1.1e}_to_{:1.1e}'.format(sol1['chiperp'],sol2['chiperp'])

            if sol1['chipar']==sol2['chipar']:
                filename = filename+'_chipar_{:1.1e}'.format(sol1['chipar'])
            else:
                filename = filename+'_chipar_from_{:1.1e}_to_{:1.1e}'.format(sol1['chipar'],sol2['chipar'])

            # if sol1['compr']==sol2['compr']:
            #     filename = filename+'_compr_{:1.1e}'.format(sol1['compr'])
            # else:
            #     filename = filename+'_compr_from_{:1.1e}_to_{:1.1e}'.format(sol1['compr'],sol2['compr'])

            if sol1['di']==sol2['di']:
                filename = filename+'_di_{:1.1e}'.format(sol1['di'])
            else:
                filename = filename+'_di_from_{:1.1e}_to_{:1.1e}'.format(sol1['di'],sol2['di'])

            if sol1['tau']==sol2['tau']:
                filename = filename+'_tau_{:1.1e}'.format(sol1['tau'])
            else:
                filename = filename+'_tau_from_{:1.1e}_to_{:1.1e}'.format(sol1['tau'],sol2['tau'])

            if sol1['m']==sol2['m']:
                filename = filename+'_m_{:1.1e}'.format(sol1['m'])
            else:
                filename = filename+'_m_from_{:1.1e}_to_{:1.1e}'.format(int(sol1['m']),int(sol2['m']))

            if sol1['n']==sol2['n']:
                filename = filename+'_n_{:1.1e}'.format(sol1['n'])
            else:
                filename = filename+'_n_from_{:1.1e}_to_{:1.1e}'.format(int(sol1['n']),int(sol2['n']))

            if sol1['N']==sol2['N']:
                filename = filename+'_N_{:d}'.format(sol1['N'])
            else:
                filename = filename+'_N_from_{:d}_to_{:d}'.format(sol1['N'],sol2['N'])
                
            if sol1['epsilon']==sol2['epsilon']:
                filename = filename+'_epsilon_{:1.1e}'.format(sol1['epsilon'])
            else:
                filename = filename+'_epsilon_from_{:1.1e}_to_{:1.1e}'.format(sol1['epsilon'],sol2['epsilon'])
                

        state = {"par_count" : self._parameter_counts,
                 "last_par_name" : self._parameter_names,
                 "omega" : self.omega,
                 "option" : self.option,
                 "solution" : self.solution,
                 "defined" : self.defined,
                 "HasOneSolution" : self.HasOneSolution,
                 "A_pressure" : self.A_pressure,
                 "A_iota" : self.A_iota,
                 "rad_pressure" : self.rad_pressure,
                 "rad_iota" : self.rad_iota,
                 "beta" : self.beta,
                 "eta" : self.eta,
                 "nu" : self.nu,
                 "chiperp" : self.chiperp,
                 "chipar" : self.chipar,
                 "compr" : self.compr,
                 "di" : self.di,
                 "tau" : self.tau,
                 "m" : self.m,
                 "n" : self.n,
                 "N" : self.N,
                 "epsilon" : self.epsilon}

        if os.path.isfile(filename+'.npz'):
            existing_filename = filename
            print '\n Saving {:}.npz\n'.format(filename)
            yn = raw_input(' This file already exists, do you want to replace it (y/n) : ')
            while (yn!='y' and yn!='n'):
                yn = raw_input(" Please answer 'y' or 'n' : ")
            if yn=='n':
                print ' \n Provide a new filename (without extension)'
                filename = raw_input(' Filename = ')
                while filename==existing_filename:
                    print ' \n The filename must be different or you will erase your data!'
                    filename = raw_input(' Filename = ')
        # Careful with the way numpy.savez saves things 
        # (we should actually use something else ideally but this works fine enough)
        np.savez(filename,solutions=self.solutions,state=[state])

    def _Load(self,filename):
        """
        Load the data in filename and reinitialize everything
        """

        Dat = np.load(filename)

        # Careful with the way numpy.savez saves things 
        # (we should actually use something else ideally but this works fine enough)
        self.solutions = list(Dat['solutions'])
        state = Dat['state'][0]

        self.A_pressure = state['A_pressure']
        self.A_iota = state['A_iota']
        self.rad_pressure = state['rad_pressure']
        self.rad_iota = state['rad_iota']
        
        self.option = state['option']

        self.beta = state['beta']
        self.eta = state['eta']
        self.nu = state['nu']
        self.chiperp = state['chiperp']
        self.chipar = state['chipar']
        self.compr = state['compr']
        self.di = state['di']
        self.tau = state['tau']
        self.m = state['m']
        self.n = state['n']
        self.N = state['N']
        self.epsilon = state['epsilon']

        self.X,self.dx = np.linspace(0,1,self.N+2,retstep=True)
        # The previous x above is no longer needed
        self.x = self.X[1:self.N+2]

        self.defined = state['defined']
        self.HasOneSolution = state['HasOneSolution']

        self.omega = state['omega']
        
        self._parameter_counts = state['par_count']
        self._parameter_names = state['last_par_name']

        self.solution = state['solution']

        self.DefinePressureIotaProfiles(self.A_pressure,self.A_iota,rad_pressure=self.rad_pressure,rad_iota=self.rad_iota)

    def _set_power_limits(self,xmini=-2,xmaxi=4,ymini=-2,ymaxi=4):
        """
        Sets the formatting of x and y labels in the current figure
        """
        
        ax=plt.gca()
        ax.xaxis.get_major_formatter().set_powerlimits((xmini,xmaxi));
        ax.yaxis.get_major_formatter().set_powerlimits((ymini,ymaxi));
        plt.draw();
        
    def _set_power_limits_x(self,xmini=-2,xmaxi=4):
        """
        Sets the formatting of x and y labels in the current figure
        """
        
        ax=plt.gca()
        ax.xaxis.get_major_formatter().set_powerlimits((xmini,xmaxi));
        plt.draw();
        
    def _set_power_limits_y(self,ymini=-2,ymaxi=4):
        """
        Sets the formatting of x and y labels in the current figure
        """
    
        ax=plt.gca()
        ax.yaxis.get_major_formatter().set_powerlimits((ymini,ymaxi));
        plt.draw();

