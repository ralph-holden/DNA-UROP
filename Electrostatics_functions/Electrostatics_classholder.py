# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 23:59:03 2024

@author: Ralph Holden
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.constants import epsilon_0, Boltzmann




# # # ELECTROSTATICS # # #
class Electrostatics:
    ''' 
    Electrostatic helical interaction as described by Kornyshev - Leikin theory
    From 'Sequence Recognition in the Pairing of DNA Duplexes', Kornyshev & Leikin, 2001, DOI: 10.1103/PhysRevLett.86.3666
    
    NOTE: uses real units of metres for length, but energy in kbT, therefore force in kbT per metre
    '''
    # constants
    eps = 80 # ~dielectric constant water
    r = 9 * 10**-10 # radius phosphate cylinder, in metres
    sigma = 16.8 # phosphate surface change density, in micro coulombs per cm^2
    sigma *= 10**-6 * (10**2)**2 # in coulombs per m^2
    f1, f2, f3 = 0.7, 0.3, 0 # fraction of counterions on; double heix minor groove, major groove, phosphate backbone
    debye = 7 * 10**-10 # Debye length (kappa^-1), in metres
    H = 34 * 10**-10 # Helical pitch, in metres
    lamb_c = 100 * 10**-10 # helical coherence length, in metres. NOTE: more recent estimate NOT from afformentioned paper
    coeffs = 16 * np.pi**2 * sigma**2 / eps # coefficients for 'a' terms, apply at end of calculations, for Eint. NOTE: requires a0 to have a prefactor of 1/2
    
    def __init__(self, homol=False, theta=0.8):
        self.homol = homol     
        
        self.theta = theta # fraction phosphate charge neutralised by adsorbed counterions
        
    def kappa(self, n):
        return np.sqrt( 1/self.debye**2  +  n**2 * (2*np.pi / self.H)**2 ) # when n=0 kappa = 1/debye
        
    def f(self, n):
        return self.f1 * self.theta  +  self.f2 * (-1)**n * self.theta  -  (1 - self.f3*self.theta) * np.cos(0.4 * n * np.pi)    

    def nu(self, n, L):
        x = n**2 * L / self.lamb_c
        return ( 1 - np.exp(-x) ) / x 
    
    def a(self, R: float) -> tuple([float, float, float]):
        '''
        Finds 'a' terms for a_0, a_1, a_2 @ R
        For a_0 term, pairwise sum from -inf to +inf approximated as -1 to 1 (including zero)
        For gen_energy_map(), R input can be array
        
        NOTE: w/out coeff factor, applied only @ Eint calculation, a0 has a prefactor of 1/2
        '''
        a0_coeff = 1/2
        a0_term1 = (1-self.theta)**2 * special.kn(0, R/self.debye ) / ( (1/self.debye**2) * special.kn(1, self.r/self.debye )**2 )
        a0_term2 = 0
        n_for_sum = 3
        for j in range(-n_for_sum,n_for_sum+1):
            for n in range(-n_for_sum,n_for_sum+1):
                a0_term2 += self.f(n)**2 / self.kappa(n)**2  *  special.kn(n-j, self.kappa(n)*R)**2 * special.ivp(j, self.kappa(n)*self.r)  /  (  special.kvp(n, self.kappa(n)*self.r)**2 * special.kvp(j, self.kappa(n)*self.r)  ) 
        a0 = a0_coeff * (a0_term1 - a0_term2)
        a1 = self.f(1)**2 / self.kappa(1)**2 * special.kn(0, self.kappa(1)*R ) / special.kvp(1, self.kappa(1)*self.r )**2
        a2 = self.f(2)**2 / self.kappa(2)**2 * special.kn(0, self.kappa(2)*R ) / special.kvp(2, self.kappa(2)*self.r )**2

        return a0, a1, a2

    def gen_energy_map(self):
        ''' 
        INPUTS:
        homol : bool , controls output for homologous and non homologous sequences, default = False.
        
        OUTPUT:
        Eint  : 2D np.array, electrostatic internal energy contribution, function of L & R.
                for homol = False -- NON homologous interaction (default)
                for homol = True  -- HOMOLOGOUS interaction
        '''
        Lmin = 0.0001 * 10**-8
        Lmax = 1 * 10**-8
        Lstep = 2 * 10**-10 # grain diameter
        Lrange = np.linspace( Lmin, Lmax, int((Lmax-Lmin)/(Lstep))+1 )
    
        Rmin = 0.2 * 10**-8
        Rmax = 0.3 * 10**-8 # same as R_cut = 0.1 * 100*10**-10
        Rstep = 0.01 * 10**-10
        Rrange = np.linspace( Rmin, Rmax, int((Rmax-Rmin)/(Rstep))+1 )
        
        Lrange, Rrange = np.meshgrid(Lrange, Rrange)
        self.Lrange = Lrange
        self.Rrange = Rrange
        
        self.a0, self.a1, self.a2 = self.a(Rrange)

        if not self.homol:
            self.Eint = self.coeffs * ( self.a0  -  self.nu(1, Lrange) * self.a1 * np.cos(np.arccos(self.a1/(4*self.a2)))  +  self.nu(2, Lrange) * self.a2 * np.cos(2*np.arccos(self.a1/(4*self.a2))) ) * Lrange
        elif self.homol:
            self.Eint = self.coeffs * ( self.a0  -  self.a1*np.cos(np.arccos(self.a1/(4*self.a2)))  +  self.a2*np.cos(2*np.arccos(self.a1/(4*self.a2))) ) * Lrange
        
        self.Eint *= (4*np.pi*epsilon_0)
        self.Eint /= (Boltzmann * 300) # get in kbT units
        self.Eint *= 10**8 # multiply by l_0 for factor of L[m] to L[l_c]

    def calc_a0l0(self, R: float) -> float:
        '''
        Finds 'a' terms for a_0 @ R
        For a_0 term, pairwise sum from -inf to +inf approximated as -1 to 1 (including zero)
        For gen_energy_map(), R input can be array
        
        NOTE: w/out coeff factor, applied only @ Eint calculation, a0 has a prefactor of 1/2
        '''
        a0_coeff = 1/2
        a0_term1 = (1-self.theta)**2 * special.kn(0, R/self.debye ) / ( (1/self.debye**2) * special.kn(1, self.r/self.debye )**2 )
        a0_term2 = 0
        n_for_sum = 3
        for j in range(-n_for_sum,n_for_sum+1):
            for n in range(-n_for_sum,n_for_sum+1):
                a0_term2 += self.f(n)**2 / self.kappa(n)**2  *  special.kn(n-j, self.kappa(n)*R)**2 * special.ivp(j, self.kappa(n)*self.r)  /  (  special.kvp(n, self.kappa(n)*self.r)**2 * special.kvp(j, self.kappa(n)*self.r)  ) 
        a0 = a0_coeff * (a0_term1 - a0_term2)
        
        a0 *= self.coeffs
        a0 *= 1e-18 # L0 coeff
        a0 /= Boltzmann

        return a0
    
    def calc_da0dR(self, R: float) -> float:
        n_for_sum = 3
        a0_coeff = 1/2
        d_a0_dR = a0_coeff * (
            # Derivative of a0_term1
            (1-self.theta)**2 * (1/self.debye) * special.kn(1, R/self.debye) / ( (1/self.debye**2) * special.kn(1, self.r/self.debye)**2 )
            
            # Derivative of a0_term2
            - sum(
                sum(
                    2 * self.f(n)**2 / self.kappa(n)**2 * 
                    special.kn(n-j, self.kappa(n)*R) * self.kappa(n) * special.kn(1, self.kappa(n)*R) * 
                    special.ivp(j, self.kappa(n)*self.r) / 
                    (special.kvp(n, self.kappa(n)*self.r)**2 * special.kvp(j, self.kappa(n)*self.r))
                    for n in range(-n_for_sum, n_for_sum+1)
                )
                for j in range(-n_for_sum, n_for_sum+1)
            )
        )
        d_a0_dR *= self.coeffs
        d_a0_dR *= 1e-18 # L0 coeff
        d_a0_dR /= Boltzmann
        return d_a0_dR

    def find_energy(self, Lindex: int, R: float):
        '''         
        INPUTS:
        Lindex: int  , index of truncated grain pairing interaction, one unit is 0.2 lamb_c. NOTE: can be a numpy array (1D)
                       NOTE: force can be str ('homol x'), changing the mode of interaction to homologous from NON homologous
                             in this case, 'x' gives the displacement in the homologous recognition funnel, L is not included as the energy is per unit length (of each grain)
        R     : float, inter-grain separation.
        
        OUTPUT:
        Energy: float, magnitude (including +-) of electrostatic interaction
        '''
        ishomol = type(Lindex) == str 

        a0, a1, a2 = self.a(R)

        SP_min_factor = np.cos(np.arccos(np.clip(a1/(4*a2), -1, 1))) if abs(a2) > 10**-10 else np.cos(np.pi)

        Lmin = 20  * 10**-10
        Lmax = 100000 * 10**-10
        Lstep = 20 * 10**-10 # grain diameter
        Lrange = np.linspace( Lmin, Lmax, int((Lmax-Lmin)/(Lstep))+1 )
        L = Lrange[Lindex-1] if not ishomol else int(Lindex.split(' ')[1])
        #L0 = 18e-8
            
        if not ishomol: 
            Eint = self.coeffs * ( a0  -  self.nu(1, L) * a1 * SP_min_factor  +  self.nu(2, L) * a2 * SP_min_factor ) * L # use L, corrected in model
            
        elif ishomol:
            Eint = self.coeffs * ( a0  -  a1 * SP_min_factor  +  a2 * SP_min_factor ) * np.exp( -L/5 ) * 0.2e-8
            
        Eint *= 9e9 # gaussian to real (J)

        return Eint
    
    def force(self, Lindex: int, R: float):
        '''         
        INPUTS:
        Lindex: int  , index of truncated grain pairing interaction, one unit is 0.2 lamb_c. NOTE: can be a numpy array (1D)
                       NOTE: force can be str ('homol x'), changing the mode of interaction to homologous from NON homologous
                             in this case, 'x' gives the displacement in the homologous recognition funnel, L is not included as the energy is per unit length (of each grain)
        R     : float, inter-grain separation.
        
        OUTPUT:
        Force  : float, magnitude (including +-) of electrostatic interaction
                 negative gradient of energy with respect to separation, R 
        '''
        h = 0.00001e-8
        ishomol = type(Lindex) == str 
        
        #Rrange = np.linspace(R-1*h , R+1*h, 3)
        #a0, a1, a2 = self.a(Rrange)
        a0_ph, a1_ph, a2_ph = self.a(R+h)
        a0_mh, a1_mh, a2_mh = self.a(R-h)

        #SP_min_factor = []
        #for i in range(len(a1)):
        #    a11 = a1[i]
        #    a22 = a2[i]
        #    SP_min_factor += [np.cos(np.arccos(np.clip(a11/(4*a22), -1, 1)))] if abs(a22) > 10**-10 else [np.cos(np.pi)] 
        #SP_min_factor = np.array(SP_min_factor)
        
        SP_min_factor_ph = np.cos(np.arccos(np.clip(a1_ph/(4*a2_ph), -1, 1))) if abs(a2_ph) > 10**-10 else np.cos(np.pi)
        SP_min_factor_mh = np.cos(np.arccos(np.clip(a1_mh/(4*a2_mh), -1, 1))) if abs(a2_mh) > 10**-10 else np.cos(np.pi) # avoid dividing by zero error

        Lmin = 20  * 10**-10
        Lmax = 100000 * 10**-10
        Lstep = 20 * 10**-10 # grain diameter
        Lrange = np.linspace( Lmin, Lmax, int((Lmax-Lmin)/(Lstep))+1 )
        L = Lrange[Lindex-1] if not ishomol else int(Lindex.split(' ')[1])
        #L0 = 18e-8 # # # needs adjustment so can be used for sims with both real and lc units # # # 
            
        if not ishomol: 
            #Eint = self.coeffs * ( a0  -  self.nu(1, L) * a1 * SP_min_factor  +  self.nu(2, L) * a2 * SP_min_factor )
            #Eint *= L0 #**Lindex
            #Force = -np.gradient(Eint)[2]
            Eint_phpL = self.coeffs * ( a0_ph  -  self.nu(1, L+h) * a1_ph * SP_min_factor_ph  +  self.nu(2, L+h) * a2_ph * SP_min_factor_ph ) * (L+h)
            Eint_phmL = self.coeffs * ( a0_ph  -  self.nu(1, L-h) * a1_ph * SP_min_factor_ph  +  self.nu(2, L-h) * a2_ph * SP_min_factor_ph ) * (L-h)
            Eint_mhpL = self.coeffs * ( a0_mh  -  self.nu(1, L+h) * a1_mh * SP_min_factor_mh  +  self.nu(2, L+h) * a2_mh * SP_min_factor_mh ) * (L+h)
            Eint_mhmL = self.coeffs * ( a0_mh  -  self.nu(1, L-h) * a1_mh * SP_min_factor_mh  +  self.nu(2, L-h) * a2_mh * SP_min_factor_mh ) * (L-h)
            Force = -1 * (Eint_phpL + Eint_mhmL - Eint_mhpL - Eint_phmL)/ ( 4*h**2 ) # -VE gradient of R,L
            
        elif ishomol:
            #Eint = self.coeffs * ( a0  -  a1 * SP_min_factor  +  a2 * SP_min_factor ) * np.exp( -L/5 )
            #Eint *= L0 #**Lindex
            #Force = -np.gradient(Eint)[2]
            
            Eint_phpL = self.coeffs * ( a0_ph  -  a1_ph * SP_min_factor_ph  +  a2_ph * SP_min_factor_ph ) * (L+h) * np.exp( -L/5 )
            Eint_phmL = self.coeffs * ( a0_ph  -  a1_ph * SP_min_factor_ph  +  a2_ph * SP_min_factor_ph ) * (L-h) * np.exp( -L/5 )
            Eint_mhpL = self.coeffs * ( a0_mh  -  a1_mh * SP_min_factor_mh  +  a2_mh * SP_min_factor_mh ) * (L+h) * np.exp( -L/5 )
            Eint_mhmL = self.coeffs * ( a0_mh  -  a1_mh * SP_min_factor_mh  +  a2_mh * SP_min_factor_mh ) * (L-h) * np.exp( -L/5 ) # 'fix' for recognition funnel
            Force = -1 * (Eint_phpL + Eint_mhmL - Eint_mhpL - Eint_phmL)/ ( 4*h**2 ) # -VE gradient of R,L
            
        Force *= 0.2e-8 # multiply by l0
        Force *= 9e9 # gaussian to SI
            
        return Force
        
    def plot_energy_map(self):
        '''Must be used after gen_energy_map()'''
        x, y = self.Lrange, self.Rrange
        z = self.Eint
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(x, y, z, cmap='viridis', alpha=0.7)
        
        fig.colorbar(surf)
        
        ax.set_xlabel('L (m)')
        ax.set_ylabel('R (m)')
        ax.set_zlabel('Eint (kbT)')
        
        plt.show()