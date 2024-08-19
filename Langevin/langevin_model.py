# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 18:11:06 2024

@author: Ralph Holden

MODEL:
    polymer bead model - Worm-like-Chain, including excluded volume and specialised dsDNA interactions
    beads correspond to 1/5 correlation length as described in Kornyshev-Leiken theory
        as such, for non homologous DNA, only one consecutive helical coherence length can have attractive charged double helix interactions
   
CODE & SIMULATION:
    Langevin dynamics (a Monte Carlo method) used to propagate each grain in the dsDNA strands
    Additional forces involved are; 
        harmonic grain springs for keeping the strand intact (artifical 'fix')
        harmonic angular springs for the worm like chain
        conditional electrostatic interactions as described in Kornyshev-Leikin theory
        + keeping inside the simulation box (confined DNA, closed simulation, artificicial 'fix' to avoid lost particles)
    Energy dependant on:
        worm like chain bending (small angle approx -> angular harmonic)
        conditional electrostatic interactions as described in Kornyshev-Leikin theory

NOTE: architecture improvement may involve changing some functions to take specific Grains as arguments,
    so that a single loop around cominations can be used for many functions - faster
"""
# imports
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from scipy import special
from scipy.constants import epsilon_0, Boltzmann

# # # UNITS # # #
kb = 1 # 1.38e-23
temp = 310.15

# # # PARAMETERS # # #
# Worm Like Chain Bending
lp = 5 # persistence length, in coherence length diameter grains of 100 Angstroms
kappab = lp * kb * temp # bending stiffness
s = 0.4 # standard distance through chain separated by one Grain
k_bend = kappab/s # Bending stiffness constant

k_spring = 300*kb  # Spring constant for bonds

# Simulation Interaction Parameters
R_cut = 0.75 # cut off distance for electrostatic interactions, SURFACE to SURFACE distance, in helical coherence lengths (100 Angstroms) therefore 7.5 nm in real units
self_interaction_limit = 5 # avoid interactions between grains in same strand
homology_set = False

# Langevin 
lamb = 0.75 # damping coefficient
dt = 0.0001 # timestep, per unit mass



# # # Aux functions # # #
class Start_position:
    
    def __init__(self, num_segments, xstart, ystart, zstart):
        # Parameters
        self.total_points = num_segments # Total number of points (adjust this value as needed)
        self.points_per_semicircle = int(round(5 * 5)) # 25 points per semicircle
        self.radius = lp / np.pi # Radius of the semicircles
        self.num_semicircles = self.total_points // self.points_per_semicircle  # Number of complete semicircles
    
        self.xstart = xstart
        self.ystart = ystart
        self.zstart = zstart
    
        
    
    def create_strand_curved(self):
        # Create arrays for x, y, and z coordinates
        self.x = []
        self.y = []
        self.z = []
        
        # Generate the snake curve 
        for n in range(self.total_points):
            # Determine the current semicircle number
            semicircle_num = n // self.points_per_semicircle
            # Determine the position within the current semicircle
            theta = (n % self.points_per_semicircle) / self.points_per_semicircle * np.pi
            m = (-1)**semicircle_num
            
            # Calculate x, y, and z based on the current semicircle
            self.x.append(self.xstart + m * self.radius * np.sin(theta))  # Rotate semicircles by 90 degrees (horizontal)
            self.y.append(self.ystart + self.radius * np.cos(theta) - semicircle_num * 2 * self.radius)  # Move down vertically
            self.z.append(0)  # All z-values are zero
        
        grains = []
        for i in range( self.total_points ):
            xi, yi, zi = self.x[i], self.y[i], self.z[i]
            grains.append( Grain( [xi, yi, zi], np.zeros(3) ) )
        
        return Strand(grains)
    
    def create_strand_straight(self):
        grains = [ Grain( np.array([self.xstart, self.ystart, self.zstart]) , np.zeros(3) ) ]
        for i in range(self.total_points-1):
            grains.append( Grain( grains[-1].position + np.array([0, 0.2, 0]), np.zeros(3) ) )
        return Strand(grains)
    
    def plot_start(self):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
    
        ax.plot(self.x, self.y, self.z, linestyle='', marker='.', markersize=10)
        ax.set_title("Snake Curve in 3D")
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
    
        plt.show()



# # # ELECTROSTATICS # # #
class Electrostatics:
    ''' 
    Electrostatic helical interaction as described by Kornyshev - Leikin theory
    From 'Sequence Recognition in the Pairing of DNA Duplexes', Kornyshev & Leikin, 2001, DOI: 10.1103/PhysRevLett.86.3666
    '''
    # constants
    eps = 80 # ~dielectric constant water, conversion to per Angstrom^-3
    r = 9 * 10**-10 # radius phosphate cylinder, in Angstroms 
    sigma = 16.8 # phosphate surface change density, in micro coulombs per cm^2
    sigma /= 10**-6 * (10**2)**2 # in coulombs per Angstrom^2
    theta = 0.8 # fraction phosphate charge neutralised by adsorbed counterions
    f1, f2, f3 = 0.7, 0.3, 0 # fraction of counterions on; double heix minor groove, major groove, phosphate backbone
    debye = 7 * 10**-10 # Debye length (kappa^-1), in Angstroms
    H = 34 * 10**-10 # Helical pitch, in Angstroms
    lamb_c = 100 * 10**-10 # helical coherence length, in Angstroms. NOTE: more recent estimate NOT from afformentioned paper
    coeffs = 16 * np.pi**2 * sigma**2 / eps # coefficients for 'a' terms, apply at end of calculations, for Eint. NOTE: requires a0 to have a prefactor of 1/2
    
    def __init__(self, homol=False):
        self.homol = homol     
        
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
        for j in range(-1,2):
            for n in range(-1,2):
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
        Lmin = 20  * 10**-10
        Lmax = 300 * 10**-10
        Lstep = 20 * 10**-10 # grain diameter
        Lrange = np.linspace( Lmin, Lmax, int((Lmax-Lmin)/(Lstep))+1 )
    
        Rmin = 0.9 * 10**-10
        Rmax = 10 * 10**-10 # same as R_cut = 0.1 * 100*10**-10
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
        self.Eint /= (Boltzmann * 300) 
        self.Eint *= 10**8 # multiply by l_0 for factor of L[m] to L[l_c]

    def find_energy(self, Lindex: int, R: float, ishomol = False):
        '''Finds energy of ONE L, R point'''
        
        Lmin = 20  * 10**-10
        Lmax = 100000 * 10**-10
        Lstep = 20 * 10**-10 # grain diameter
        Lrange = np.linspace( Lmin, Lmax, int((Lmax-Lmin)/(Lstep))+1 )
        L = Lrange[Lindex-1]
        
        a0, a1, a2 = self.a(R)

        SP_min_factor = np.cos(np.arccos(np.clip(a1/(4*a2), -1, 1))) if abs(a2) > 10**-10 else np.cos(np.pi) # avoid dividing by zero error
        
        if not ishomol: 
            Eint = self.coeffs * ( a0  -  self.nu(1, L) * a1 * SP_min_factor  +  self.nu(2, L) * a2 * SP_min_factor ) * L
        elif ishomol:
            Eint = self.coeffs * ( a0  -  a1 * SP_min_factor  +  a2 * SP_min_factor ) * 1
            
        Eint *= (4*np.pi*epsilon_0)
        Eint /= (Boltzmann * 300) # from gaussian to in kbT units
        # Eint *= 10**8 # multiply by l_0 for factor of L[m] to L[l_c]
        
        return Eint
    
    def find_energy_fc(self, L: float, R: float, ishomol = False):
        '''
        Finds energy of ONE L, R point with fully continuous (fc) inputs 
        For use in find_force(), where dE/dRdL required
        Additional boolean input from force() function, default NON homologous
        '''
        a0, a1, a2 = self.a(R)
        
        SP_min_factor = np.cos(np.arccos(np.clip(a1/(4*a2), -1, 1))) if abs(a2) > 10**-10 else np.cos(np.pi) # avoid dividing by zero error
        
        if not ishomol: 
            Eint = self.coeffs * ( a0  -  self.nu(1, L) * a1 * SP_min_factor  +  self.nu(2, L) * a2 * SP_min_factor ) * L
        elif ishomol:
            Eint = self.coeffs * ( a0  -  a1 * SP_min_factor  +  a2 * SP_min_factor ) * 1
        
        Eint *= (4*np.pi*epsilon_0)
        Eint /= (Boltzmann * 300) # from gaussian to in kbT units
        # Eint *= 10**8 # for factor of L[m] to L[l_c]
        
        return Eint

    def force(self, Lindex: int, R: float):
        '''         
        INPUTS:
        Lindex: int  , index of truncated grain pairing interaction, one unit is 0.2 lamb_c. NOTE: can be a numpy array (1D)
        R     : float, inter-grain separation.
        
        OUTPUT:
        Force  : float, magnitude (including +-) of electrostatic interaction
                 negative gradient of energy with respect to separation, R 
        '''
        ishomol = type(Lindex) == str # for individual interaction, allows for non homologous interactions within homologous system
        
        Lmin = 20  * 10**-10
        Lmax = 100000 * 10**-10
        Lstep = 20 * 10**-10 # grain diameter
        Lrange = np.linspace( Lmin, Lmax, int((Lmax-Lmin)/(Lstep))+1 )
        L = Lrange[Lindex-1] if ishomol else 1
        
        h = 0.0001 * 10**-10 # for differentiation by first principles
        # mixed differentiation by first principles, NOTE: could use analytical derivative to save computational cost
        if not ishomol:
            dE_dRdL = ( self.find_energy_fc(L-h, R-h) + self.find_energy_fc(L+h, R+h) - self.find_energy_fc(L-h, R+h) - self.find_energy_fc(L+h, R-h) ) / ( 4*h**2 )
            return -1*dE_dRdL
        if ishomol:
            dE_dR = ( self.find_energy(Lindex, R+h, ishomol) - self.find_energy(Lindex, R, ishomol) ) / h
            return -1*dE_dR
        
    def plot_energy_map(self):
        '''Must be used after gen_energy_map()'''
        x, y = self.Lrange, self.Rrange
        z = self.Eint
        
        # Create a new figure for the 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface
        surf = ax.plot_surface(x, y, z, cmap='viridis', alpha=0.7)
        
        # Add a color bar which maps values to colors
        fig.colorbar(surf)
        
        # limits
        #ax.set_xlim([np.min(x), np.max(x)])
        #ax.set_ylim([np.min(y), np.max(y)])
        
        # Set labels
        ax.set_xlabel('L (m)')
        ax.set_ylabel('R (m)')
        ax.set_zlabel('Eint (kbT)')
        
        # Show the plot
        plt.show()

# Pairing interaction energy
elstats = Electrostatics(homol = homology_set)



class Grain():
    def __init__(self, position: np.array, velocity: np.array, radius = 0.1):
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.radius = radius
        self.mass = 1

    def update_velocity(self, force, dt):
        # Update velocity based on the applied force
        self.velocity += force / self.mass * dt

    def update_position(self, dt):
        # Update position based on the velocity
        self.position += self.velocity * dt
        
    def copy(self):
        return Grain(self.position, self.velocity, self.radius)
    
    
    
class Strand:
    
    def __init__(self, grains):
        self.dnastr = grains
        self.num_segments = len(grains)
        self.isinteract = np.zeros(self.num_segments)==np.ones(self.num_segments)
        self.interactions = []
        
        # Gives list indexes that can break the self interaction lower limit, by interacting ACROSS the strands
        self.cross_list = [] # defined here as to avoid repetitions. 
        for i in range(self.num_segments - self_interaction_limit, self.num_segments):
            for j in range(i, i+self_interaction_limit+1):
                if j >= self.num_segments:
                    self.cross_list.append([i,j]) 
                    
        # Gives list of repetitions ***OUTDATED***
        self.repetitions = [ [],[] ]
        
    def copy(self):
        return (Strand(grains = self.dnastr))
    
    # bond springs, to hold chain together
    def f_bond(self):
        for i in range(self.num_segments - 1):
            # Vector between consecutive grains
            delta = self.dnastr[i+1].position - self.dnastr[i].position
            distance = np.linalg.norm(delta)
            force_magnitude = k_spring * (distance - 2*self.dnastr[0].radius)
            force_direction = delta / distance if distance != 0 else np.zeros(3)
            force = force_magnitude * force_direction
            self.dnastr[i].update_velocity(force, 1)
            self.dnastr[i+1].update_velocity(-force, 1)
                
    # WLC bending energies
    def f_wlc(self):
        for i in range(1, self.num_segments - 1):
            # Vectors between adjacent grains
            r1 = self.dnastr[i-1].position - self.dnastr[i].position
            r2 = self.dnastr[i+1].position - self.dnastr[i].position
            r1_norm = np.linalg.norm(r1)
            r2_norm = np.linalg.norm(r2)
            # Cosine of the angle between r1 and r2
            cos_theta = np.dot(r1, r2) / (r1_norm * r2_norm)
            theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            if np.isclose(theta,np.pi):
                continue # angle equivalent to 0 therefore no torque required
            torque_magnitude = -k_bend * (theta - np.pi) 
            torque_direction = r1+r2
            torque_direction /= np.linalg.norm(torque_direction) if np.linalg.norm(torque_direction) != 0 else 1
            torque = torque_magnitude * torque_direction
            self.dnastr[i-1].update_velocity(-torque, dt)
            self.dnastr[i].update_velocity(torque, dt)
            self.dnastr[i+1].update_velocity(-torque, dt)
        
    def find_angle(self, seg_index):
        p1 = self.dnastr[seg_index-1].position
        p2 = self.dnastr[seg_index].position
        p3 = self.dnastr[seg_index+1].position
        
        # colinear vectors
        if np.isclose(p3-p2,p2-p1).all():
            return 0
        # return 180 - angle
        return np.pi - np.arccos(np.dot(p3-p2,p1-p2) / (np.linalg.norm(p3-p2)*np.linalg.norm(p1-p2) ) )
        
    def find_interactions(self, other) -> list:
        '''
        Generates from 0th Bead, electrostatic interaction counted for whole Strand
        Does for BOTH strands, interactions attribute for StrandA contains ALL information for BOTH strands
        '''
        # reset interactions
        self.interactions = [ [[],[]] ]
        self.repetitions  = [ [[],[]] ]
        
        # loop through all combinations
        for i, j in combinations(range(len(self.dnastr+other.dnastr)),2):
            if abs(i-j) <= self_interaction_limit and [i,j] not in self.cross_list:
                continue # skips particular i, j if a close self interaction, avoiding application across strands
            igrain = self.dnastr[i] if i<self.num_segments else other.dnastr[i-self.num_segments] 
            jgrain = self.dnastr[j] if j<self.num_segments else other.dnastr[j-self.num_segments] # use correct strand
            R = jgrain.position - igrain.position
            R_norm = np.linalg.norm(R) - 0.2 # get surface - surface distance
            
            # if interaction under cutoff distance
            if R_norm < R_cut and R_norm != 0: 
                # find IDs
                idnti = 's'+str(i) if i < self.num_segments else 'o'+str(i-self.num_segments)
                idntj = 's'+str(j) if j < self.num_segments else 'o'+str(j-self.num_segments)
                
                # add to interactions
                # check if add to any existing 'islands'
                island = self.find_island(idnti, idntj)
                island = 0 if len(self.interactions[0][0]) == 0 else island # add to first island
                
                # check if BOTH IDs are in ONE other island+repetitions, if so, to ignore 
                ignore_interaction = self.check_repetition(other, island, idnti, idntj, R_norm)
                
                # add ID and R to island (in interactions attribute), and any possible repetitions
                if not ignore_interaction:
                    if type(island) == int: 
                        self.interactions[island][0].append(idnti+' '+idntj)
                        self.interactions[island][1].append(R_norm)
                    elif island == 'new':
                        self.interactions.append( [[],[]] ) # create new island
                        self.interactions[-1]    [0].append(idnti+' '+idntj)
                        self.interactions[-1]    [1].append(R_norm)
                #self.update_repetitions(other, island, idnti, idntj, R_norm)
                
    def find_island(self, idnti, idntj):
        # check if add to any existing 'islands'
        # create list of possible configurations for an 'island'
        check_island_configurations = []
        for stepi in range(-self_interaction_limit,self_interaction_limit):
            for stepj in range(-self_interaction_limit,self_interaction_limit):
                if int(idnti[1:])+stepi >= self.num_segments or int(idnti[1:])+stepi < 0 or int(idntj[1:])+stepj >= self.num_segments or int(idntj[1:])+stepj < 0:
                    continue
                check_island_configurations += [idnti[0]+str( int(idnti[1:])+stepi ) + ' ' +  idntj[0]+str( int(idntj[1:])+stepj )]
        # check possible configurations against existing islands
        for n in range(len(self.interactions)):
            for check_idnt in check_island_configurations:
                if check_idnt in self.interactions[n][0]:   
                    return n
        return 'new' 
    
    def check_repetition(self, other, island, idnti, idntj, R_norm) -> bool:
        ignore_interaction = False
        if type(island) == int:
            is_idnti, is_idntj = False, False
            for i in range(len( self.interactions[island][0] )):
                is_idnti = True if idnti in self.interactions[island][0][i].split(' ') else is_idnti
                is_idntj = True if idntj in self.interactions[island][0][i].split(' ') else is_idntj
                #R_compare = self.interactions[island][1][i]
            if is_idnti or is_idntj:
                ignore_interaction = True #if R_compare > R_norm else ignore_interaction
        return ignore_interaction
    
    def update_repetitions(self, other, island, idnti, idntj, R_norm):
        ''' function, along with repetitions attribute no longer used '''
        if island == 'new':
            self.repetitions.append( [[],[]] )
            island = -1
        self.repetitions[island][0].append(idnti+' '+idntj)
        self.repetitions[island][1].append(R_norm)
        for n in range(self_interaction_limit):
            for stepi, stepj in [ [-n,0], [0,-n], [n,0], [0,n] ]: # can change this for larger loops until interaction 'reset'
                if int(idnti[1:])+stepi >= self.num_segments or int(idnti[1:])+stepi < 0 or int(idntj[1:])+stepj >= self.num_segments or int(idntj[1:])+stepj < 0:
                    continue
                self.repetitions[island][0] += [idnti[0]+str( int(idnti[1:])+stepi ) + ' ' +  idntj[0]+str( int(idntj[1:])+stepj )]
                
                g1 = self.dnastr[ int(idnti[1:])+stepi ] if idnti[0] == 's' else other.dnastr[ int(idnti[1:])+stepi ]
                g2 = self.dnastr[ int(idntj[1:])+stepj ] if idntj[0] == 's' else other.dnastr[ int(idntj[1:])+stepj ]
                self.repetitions[island][1] += [ np.linalg.norm( g1.position - g2.position ) - 0.2 ]
                
    def assign_L(self, other):
        for isle in self.interactions:
            # arrange in order ?
            if isle != [[],[]]:
                leading = np.argmin(isle[1])
                Llist = [1]
                for i in range(2, leading+2): # backwards from leading-1 to index 0
                    Llist = [i] + Llist
                for i in range(2, len(isle[0])-leading+1): # forwards from leading + 1
                    Llist += [i]
                if homology_set:
                    for inter, index in enumerate(isle[0]):
                        if inter.split(' ')[0][1:] == inter.split(' ')[1][1:] :
                            Llist[index] = 'homol'
                isle.append(Llist)
            else:
                isle.append([])
                
            
            
    def f_elstat(self, other):
        ''' 
        NOTE: does not yet account for NON homologous interactions for HOMOLOGOUS strands
        '''
        self.find_interactions(other)
        self.assign_L(other)
        
        # must only take MOST SIGNIFICANT (closest) interaction for each grain in an interacting 'island'

        for isle in self.interactions:
            for i in range(len(isle[0])):
                felstat = elstats.force(isle[2][i], isle[1][i]*10**-8) # change to metres
                # identify relevant grains
                idnt1 = isle[0][i].split(' ')[0]
                idnt2 = isle[0][i].split(' ')[1]
                grain1 = self.dnastr[ int(idnt1[1:]) ] if idnt1[0]=='s' else other.dnastr[ int(idnt1[1:]) ]
                grain2 = self.dnastr[ int(idnt2[1:]) ] if idnt2[0]=='s' else other.dnastr[ int(idnt2[1:]) ]
                fvec = grain2.position - grain1.position
                fvec /= np.linalg.norm(fvec) if np.linalg.norm(fvec) != 0 else 1
                grain1.update_velocity(+1*felstat*fvec,dt)
                grain2.update_velocity(-1*felstat*fvec,dt)
        
    # for energies, not including bond springs or translational energy
    def eng_elstat(self, other, homol=False):
        '''
        Does electrostatic energy of BOTH strands
        In each step, use after f_elstat() so gen functions do not have to be repeated, homol argument in f_elstat() defines homologicity
        '''
        energy = 0
        for isle in self.interactions:
            if isle[1] == [] or isle[2] == []:
                continue
            for n in range(len( isle[1] )):
                g_R = isle[1][n]
                g_L = isle[2][n]
                energy +=  elstats.find_energy(g_L, g_R) - elstats.find_energy(g_L-1, g_R) # remove 'built-up' energy over L w/ different R
        return energy
        
    def eng_elastic(self) -> float:
        '''
        Energy term for bending of DNA strand from straight
        Uses small angle approx so finer coarse graining more accurate
        '''
        energy = 0
        for seg_index in range(1,self.num_segments-1):
            angle = self.find_angle(seg_index)
            energy += 1/2 * k_bend *  angle**2
        return energy / (kb*300) # give energy in kbT units
    
    # for data
    def find_centremass(self):
        av = 0
        for g in self.dnastr:
            av += g.position
        return av
    
    
    
class Simulation:
    def __init__(self, StrandA: Strand, StrandB: Strand, boxlims: np.array):
        self.StrandA = StrandA
        self.StrandB = StrandB
        
        self.boxlims = boxlims
    
        # data
        self.trajectoryA = []
        self.trajectoryB = []
        self.pair_counts = []
        self.energies = []
        self.endtoends = []
        self.centremass = []
        self.n_islands = []
        self.av_R_islands = []
        self.av_L_islands = [] 
        self.av_sep_islands = []
        self.interactions_traj = []
        #self.record()

    def run_step(self):
        self.apply_langevin_dynamics()
        self.StrandA.f_bond(), self.StrandB.f_bond()
        self.StrandA.f_wlc(), self.StrandB.f_wlc() 
        self.StrandA.f_elstat(self.StrandB)
        self.apply_box()
        self.update_positions()
        self.record()
        
    def apply_langevin_dynamics(self):
        for grain in self.StrandA.dnastr + self.StrandB.dnastr:
            # Drag force
            damping_force = -lamb * grain.velocity # with lamb = 1, zeroes previous velocity -> Brownian
            # apply drag, using 'trick' dt=1 to rescale velocity fully
            grain.update_velocity(damping_force, 1)
            # Random thermal force
            random_force = np.random.normal(0, np.sqrt(2 * lamb * kb * temp / dt), size=3) # mass set to unity
            # Damping force
            grain.update_velocity(random_force,dt)
            
    def apply_box(self):
        returning_force = 2000
        for grain in self.StrandA.dnastr + self.StrandB.dnastr:
            for i in range(3):
                if grain.position[i] + grain.radius > self.boxlims[i]:
                    grain.update_velocity(-1*returning_force, dt) 
                if grain.position[i] - grain.radius < self.boxlims[i]:
                    grain.update_velocity(+1*returning_force, dt)
       
    def update_positions(self):
        for grain in self.StrandA.dnastr + self.StrandB.dnastr:
            grain.update_position(dt)
            
    # for data analysis
    def record(self):
        
        new_trajA = []
        for grain in self.StrandA.dnastr:
            new_trajA.append(np.array([grain.position[0], grain.position[1], grain.position[2]]))
        self.trajectoryA.append(new_trajA)
        
        new_trajB = []
        for grain in self.StrandB.dnastr:
            new_trajB.append(np.array([grain.position[0], grain.position[1], grain.position[2]]))
        self.trajectoryB.append(new_trajB)
        
        self.energies.append(self.find_energy())
        
        self.endtoends.append(self.endtoend(-1))
        
        #self.centremass.append([self.StrandA.find_centremass(),self.StrandB.find_centremass()])
        
        totpair, selfpair, n_islands, av_R_islands, av_L_islands, av_sep_islands = self.islands_data()
        self.pair_counts.append([totpair, selfpair])
        self.n_islands.append(n_islands)
        self.av_R_islands.append(av_R_islands)
        self.av_L_islands.append(av_L_islands)
        self.av_sep_islands.append(av_sep_islands)
        
        if self.StrandA.interactions != [[[], [], []]]:
            self.interactions_traj.append(self.StrandA.interactions)
    
    def endtoend(self, tindex):
        endtoendA = np.linalg.norm(self.trajectoryA[tindex][0] - self.trajectoryA[tindex][-1]) + 0.2
        endtoendB = np.linalg.norm(self.trajectoryB[tindex][0] - self.trajectoryB[tindex][-1]) + 0.2 # account for size of particle
        return endtoendA, endtoendB
    
    def islands_data(self):
        '''
        Finds ALL data concerning interacting 'islands'
        
        INPUTS: (implicit)
            self.StrandA.interactions : list, contains information of ALL interacting pairs
                                        list[0] - ID strings of each interaction
                                        list[1] - R, surface-surface distance between each interacting pairs
                                        list[2] - L, contains length index for each interacting pair, w/ 'LEADING' & Deraguin approx.
        
        OUTPUTS:
            count_total      : int  , total number of interacting pairs
            count_self_pairs : int  , number of pairs interacting within a strand
            n_islands        : int  , the number of separate interacting islands
            av_R_islands     : float, mean separation between pairs across all islands,    in units of l_c
            av_l_islands     : float, mean length (through strand)  across all islands,    in units of l_c
            av_sep_islands   : float, mean length (through strand) separating all islands, in units of l_c
            
            NOTE: for empty interactions list, output is 
                  0, 0, 0, None, 0, None
        '''
        if self.StrandA.interactions == [[[], [], []]]:
            return 0, 0, 0, None, 0, None
        
        n_islands = len(self.StrandA.interactions) # number of loops / islands
        
        count_total = 0
        count_pairs = 0
        av_R_islands = 0
        start_end_g = [],[]
        
        for isle in self.StrandA.interactions:
            
            count_total += len(isle[0])
            for i in range(len(isle[0])):
                count_pairs += 1 if isle[0][i][0]==isle[0][i][4] else 0
            
            av_R_islands += np.mean( isle[1] )
            
            for i in [0,1]:
                for j in [0,-1]:
                    start_end_g[i].append( int( isle[0][j] . split(' ')[i][1:] ) )
        
        av_R_islands /= n_islands
        
        L_islands    = np.diff( start_end_g ) [:,  ::2] + 1
        sep_islands  = np.diff( start_end_g ) [:, 1::2] - 1 if len( np.diff( start_end_g ) [:, 1::2] ) != 0 else None
        av_L_islands   = np.mean( L_islands   ) * 0.2
        av_sep_islands = np.mean( sep_islands ) * 0.2 if sep_islands.any() != None else None # convert to units of coherence lengths
        
        return count_total, count_pairs, n_islands, av_R_islands, av_L_islands, av_sep_islands
          
    def find_energy(self):
        '''Includes electrostatic and WLC bending energies ONLY'''
        return self.StrandA.eng_elastic() + self.StrandB.eng_elastic() + self.StrandA.eng_elstat(self.StrandB)
