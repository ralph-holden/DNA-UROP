# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:06:51 2024

@author: 44775 Ralph Holden

MODEL:
    polymer bead model - Worm-like-Chain, including excluded volume and specialised dsDNA interactions
    beads correspond to 1/5 correlation length as described in Kornyshev-Leiken theory
        as such, for non homologous DNA, only one consecutive helical coherence length can have attractive charged double helix interactions
   
CODE & SIMULATION:
    Metropolis algorithm (a Monte Carlo method) used to propagate DNA strands
    Each Monte Carlo step shifts bead angle along entire dsDNA strand
    Additional requirements for a random move are; 
        excluded volume interactions
        keeping the strand intact
        and keeping inside the simulation box (confined DNA, closed simulation)
    Energy dependant on:
        worm like chain bending (small angle approx -> angular harmonic)
        conditional electrostatic interactions as described in Kornyshev-Leikin theory
"""
# # # IMPORTS # # #
import numpy as np
from scipy import special
from scipy.constants import epsilon_0, Boltzmann
import matplotlib.pyplot as plt
from typing import Tuple
from itertools import combinations



# # # UNITS # # #
kb = 1
temp = 310.15



# # # PARAMETERS # # #
# Settings
homology_set = False # False -> NON homologous
catch_set = False # requirement that number of pairs cannot decrease
angle_step = np.pi / 360 / 50 # maximum angle change between each grain (theta and phi)

# Worm Like Chain Bending
lp = 5 # persistence length, in coherence length diameter grains of 100 Angstroms
# NOTE: lp for specific temperature, expect to decrease w/ temperature increase ?
kappab = lp * kb * temp # bending stiffness
s = 0.4 # standard distance through chain separated by one Grain
k_bend = kappab/s # Bending stiffness constant

# Simulation Interaction Parameters
R_cut = 0.75 # cut off distance for electrostatic interactions, SURFACE to SURFACE distance, in helical coherence lengths (100 Angstroms)
self_interaction_limit = 5 # avoid interactions between grains in same strand



# # # aux functions # # #
# function to generate DNA strand
class Start_position:
    
    def __init__(self, num_segments, xstart, ystart, zstart):
        # Parameters
        self.total_points = num_segments # Total number of points (adjust this value as needed)
        self.points_per_semicircle = int(round(5*5)) # 25 points per semicircle
        self.radius = lp / ( np.pi ) # Radius of the semicircles
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
            
        dnastr = []
        for i in range( self.total_points ):
            xi, yi, zi = self.x[i], self.y[i], self.z[i]
            dnastr.append( Bead( Vector ( xi, yi, zi ) ) )
        
        return Strand(self.total_points, dnastr)
    
    def create_strand_straight(self):
        dnastr = []
        dnastr.append(Bead(Vector(self.xstart, self.ystart, self.zstart)))
        for seg in range(self.total_points - 1):
            new_position = dnastr[-1].position + Vector(0, 0.2, 0)
            dnastr.append(Bead(Vector(new_position.x, new_position.y, new_position.z)))
        return Strand(self.total_points, dnastr)
    
    def plot_start(self):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
    
        ax.plot(self.x, self.y, self.z, linestyle='', marker='.', markersize=10)
        ax.set_title("Snake Curve in 3D")
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
    
        plt.show()
        


# # # vector class # # #
# for maths
class Vector():
    '''3D vectors'''

    def __init__(self,i1,i2,i3):
        '''Initialise vectors with x and y and z coordinates'''
        self.x = i1
        self.y = i2
        self.z = i3
        self.arr = np.array([self.x,self.y,self.z])

    def __add__(self, other):
        '''Use + sign to implement vector addition'''
        return (Vector(self.x+other.x,self.y+other.y,self.z+other.z))

    def __sub__(self, other):
        '''Use - sign to implement vector "subtraction"'''
        return (Vector(self.x-other.x,self.y-other.y,self.z-other.z))

    def __mul__(self, number: float):
        '''Use * sign to multiply a vector by a scaler on the left'''
        return Vector(self.x*number,self.y*number,self.z*number)

    def __rmul__(self, number):
        '''Use * sign to multiply a vector by a scaler on the right'''
        return Vector(self.x*number,self.y*number,self.z*number)

    def __truediv__(self, number):
        '''Use / to multiply a vector by the inverse of a number'''
        return Vector(self.x/number,self.y/number,self.z/number)

    def __repr__(self):
        '''Represent a vector by a string of 3 coordinates separated by a space'''
        return '{x} {y} {z}'.format(x=self.x, y=self.y, z=self.z)

    def copy(self):
        '''Create a new object which is a copy of the current.'''
        return Vector(self.x,self.y,self.z)

    def dot(self, other):
        '''Calculate the dot product between two 3D vectors'''
        return self.x*other.x + self.y*other.y + self.z*other.z

    def norm(self):
        '''Calculate the norm of the 3D vector'''
        return (self.x**2+self.y**2+self.z**2)**0.5
    
    def angle(self, other):
        '''Calculate the angle between 2 vectors using the dot product'''
        return np.arccos(self.dot(other) / (self.norm()*other.norm()))
    
    def cartesian_to_spherical(self):
        """
        Convert Cartesian coordinates to spherical coordinates.
        
        Parameters:
        x (float): The x-coordinate in Cartesian coordinates.
        y (float): The y-coordinate in Cartesian coordinates.
        z (float): The z-coordinate in Cartesian coordinates.
        
        Returns:
        tuple: A tuple containing the spherical coordinates (r, theta, phi).
        """
        #r = np.sqrt(x**2 + y**2 + z**2)
        r = np.linalg.norm([self.x,self.y,self.z]) # causing errors with smaller box sizes??
        theta = np.arctan2(self.y, self.x)
        phi = np.arccos(self.z / r)
        return r, theta, phi

    def spherical_to_cartesian(self, r, theta, phi):
        """
        Convert spherical coordinates to Cartesian coordinates.
        
        Parameters:
        r (float or np.ndarray): The radius in spherical coordinates.
        theta (float or np.ndarray): The azimuthal angle in spherical coordinates (in radians).
        phi (float or np.ndarray): The polar angle in spherical coordinates (in radians).
        
        Returns:
        tuple: A tuple containing the Cartesian coordinates (x, y, z).
               If r, theta, and phi are arrays, x, y, and z will also be arrays.
        """
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return Vector(x, y, z)



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



# # # bead class # # #
class Bead: 
    '''
    Each individual element of the DNA strand
    Diameter is 1/5 helical coherence length
    '''
    def __init__(self, position: Vector):
        self.position = position
        self.radius = 0.1
        
    def overlap(self, other) -> Tuple[bool, float]: 
        inter_vector = self.position.arr - other.position.arr
        min_approach = self.radius + other.radius + 0.01 # tolerance to account for rounding errors that could stall the simulation
        dist = np.linalg.norm(inter_vector)
        return dist <= min_approach, dist
    
    def inter_vector(self, other) -> float: 
        intervec = other.position.arr - self.position.arr
        return intervec
    
    def copy(self):
        return Bead(Vector(self.position.x,self.position.y,self.position.z))



# # # strand class # # #
class Strand:
    '''
    Each dsDNA strand involved in the simulation
    Worm-Like-Chain bead angles
    Kornyshev-Leiken electrostatic interactions
    '''
    
    def __init__(self, num_segments: int, dnastr: list):
        
        self.num_segments = num_segments
        self.dnastr = dnastr
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
        '''Create a new object which is a copy of the current.'''
        #Strandnew = Strand(self.num_segments, self.start_position)
        # Strandnew.dnastr = self.dnastr # make sure new DNA strand is up to date
        newdnastr = [] #List.empty_list(BeadType)
        for i in range(self.num_segments):
            seg = self.dnastr[i]
            newdnastr.append( Bead( Vector(seg.position.x,seg.position.y,seg.position.z)))
        return (Strand(self.num_segments, newdnastr)) 
    
    # checks for simulation conditions
    def check_inbox(self, boxlims) -> bool:
        ''' 
        Outputs boolean for if the segment of the proposed chain is still within the simulation box,
        as per confinement
        True if within box, False if outside of box
        '''
        seg_rad = self.dnastr[0].radius
        for seg in self.dnastr:
            if abs(seg.position.x)+seg_rad > boxlims.x or abs(seg.position.y)+seg_rad > boxlims.y or abs(seg.position.z)+seg_rad > boxlims.z:
                return False
        return True 
    
    def count_all(self, strand):
        '''
        Discounts immediate neighbours from pairs
        Must be run only after gen_interactivity
        '''
        count_total = 0
        count_self_pairs = 0
        if strand.interactions == [[[], [], []]]:
            return 0, 0
        for isle in strand.interactions:
            count_total += len(isle[0])
            for i in range(len(isle[0])):
                count_self_pairs += 1 if isle[0][i][0]==isle[0][i][4] else 0
        return count_total, count_self_pairs  
    
    def check_count_increase(self, strand) -> bool:
        '''
        Checks that the TOTAL number of pairs increases in a Monte Carlo move
        Incorporated into the montecarlostep when 'catch = True'
        
        NOTE: explicit argument for strand MUST BE that of the 'interactions' attribute holder
        '''
        return self.count_all(strand)[0] >= self.count_all(self)[0]
    
    # electrostatic interactions
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
            R_norm = np.linalg.norm(R.arr) - 0.2 # get surface - surface distance
            
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
        
    def eng_elstat(self, other, homol=False):
        '''
        Does electrostatic energy of BOTH strands
        '''
        self.find_interactions(other)
        self.assign_L(other)
        energy = 0
        for isle in self.interactions:
            if isle[1] == [] or isle[2] == []:
                continue
            for n in range(len( isle[1] )):
                g_R = isle[1][n]
                g_L = isle[2][n]
                energy +=  elstats.find_energy(g_L, g_R) - elstats.find_energy(g_L-1, g_R) # remove 'built-up' energy over L w/ different R
        return energy

    # elastic chain
    def find_angle(self, seg_index):
        p1 = self.dnastr[seg_index-1].position.arr
        p2 = self.dnastr[seg_index].position.arr
        p3 = self.dnastr[seg_index+1].position.arr
        
        # Check if the points are collinear by calculating the area of the triangle formed by the points
        # If the area is zero, the points are collinear
        if np.isclose(p3-p2,p2-p1).all():
            return 0
        
        return abs(np.arccos(np.dot(p3-p2,p1-p2) / (np.linalg.norm(p3-p2)*np.linalg.norm(p1-p2) ) ) - np.pi)
    
    def eng_elastic(self) -> float:
        '''
        Energy term for bending of DNA strand from straight
        Uses small angle approx so finer coarse graining more accurate
        '''
        energy = 0
        for seg_index in range(1,self.num_segments-1):
            angle = self.find_angle(seg_index)
            energy += kappab / (2*s) *  angle**2
        return energy / (kb*300) # give in kbT units
    
    # for MC step
    def calc_arc(self, selfindex: int, otherindex: int, dtheta: float, dphi: float) -> Vector:
        '''
        Gives the displacement of the bead for an MC bend
        Theta and thi must be in radians
        '''
        inter_vec = self.dnastr[otherindex].position - self.dnastr[selfindex].position
        r, theta, phi = inter_vec.cartesian_to_spherical()
        theta, phi = theta + dtheta, phi + dphi 
        return self.dnastr[selfindex].position + inter_vec.spherical_to_cartesian(r, theta, phi)
       
    def propose_change(self, seg_index: int, forward = True):
    
        prop_Strand = self.copy()
        
        rand_theta = np.random.random()* angle_step * [-1,1][np.random.randint(2)]
        rand_phi =   np.random.random()* angle_step * [-1,1][np.random.randint(2)]
        # shift every subsequent bead, NOTE: not applicable for final bead
        if forward:
            for nextseg in range(seg_index+1, self.num_segments): # bends down one direction of chain
                prop_Strand.dnastr[nextseg].position = prop_Strand.calc_arc(seg_index, nextseg, rand_theta, rand_phi)
        elif not forward:
            for nextseg in range(seg_index-1, -1, -1):
                prop_Strand.dnastr[nextseg].position = prop_Strand.calc_arc(seg_index, nextseg, rand_theta, rand_phi)
        return prop_Strand
        
    def propose_change_whole(self):
        # random initial segment in middle 3/5 of DNA strand
        random_start_index = np.random.randint(int(self.num_segments/10),int(9*self.num_segments/10))
        # in current model, do not need to propose a change to the starting segment
        # make copy for first time, then after, update that
        # going forwards, updating entire rest of strand each time
        prop_Strand = self.propose_change(random_start_index, forward = True) # first bend
        for seg_index in range(random_start_index+1, self.num_segments-1): # again, final bead cannot bend a further
            prop_Strand = prop_Strand.propose_change(seg_index, forward = True)
        for seg_index in range(random_start_index+1, 0, -1): # again, final bead cannot bend a further
            prop_Strand = prop_Strand.propose_change(seg_index, forward = False)
        return prop_Strand
    
    # for data
    def find_centremass(self):
        av = 0
        for g in self.dnastr:
            av += g.position.arr
        return av
   
    

class Simulation:
    '''
    Runs simulation of two dsDNA strands
    Records data
    '''
    nsteps = 0
    mctime = 0.0
    
    def __init__(self, boxlims: Vector, StrandA: Strand, StrandB: Strand):
        self.boxlims = boxlims # boxlims a Vector
        
        self.StrandA = StrandA
        self.StrandB = StrandB
        
        self.energy = self.StrandA.eng_elastic() + self.StrandB.eng_elastic() + self.StrandA.eng_elstat(self.StrandB)
    
        self.trajectoryA = []
        self.trajectoryB = []
        self.pair_counts = []
        self.eng_traj = []
        self.centremass = []
        self.n_islands = []
        self.av_R_islands = []
        self.av_L_islands = [] 
        self.av_sep_islands = []
        self.interactions_traj = []
        self.save_trajectory()
        
    def montecarlostep(self):
        #prop_StrandA, prop_StrandB = self.StrandA.propose_change_both_whole()
        prop_StrandA = self.StrandA.propose_change_whole()
        prop_StrandB = self.StrandB.propose_change_whole()
                
        # find valid configuration, need to wait for entire strand to change before excvol and inbox can fairly be applied
        prop_StrandA, prop_StrandB = self.retry(prop_StrandA, prop_StrandB, catch = catch_set)
        
        # calculate deltaE 
        prev_energy = self.energy 
        prop_energy = prop_StrandA.eng_elastic() + prop_StrandB.eng_elastic() + prop_StrandA.eng_elstat(prop_StrandB)
        deltaE = prop_energy - prev_energy
        #print('','Delta Energy: ',deltaE)
    
        if deltaE <= 0: # assign new string change, which has already 'passed' conditions from proposal 
            self.StrandA, self.StrandB = prop_StrandA, prop_StrandB
            self.energy = prop_energy
            self.mctime += 0.0 # assign energy, strings and trajectories
            #print('','-VE energy',
            #'','ACCEPTED')
    
        elif deltaE >= 0:
            random_factor = np.random.random()
            boltzmann_factor = np.e**(-1*deltaE/(1)) # deltaE in kb units
            if random_factor < boltzmann_factor: # assign new string change
                self.StrandA, self.StrandB = prop_StrandA, prop_StrandB
                self.energy = prop_energy 
                self.mctime += 0.0 # assign energy, strings and trajectories
                #print('','random factor:',random_factor,
                #'','boltz  factor:',boltzmann_factor,
                #'','ACCEPTED')
            #else:
                #print('','random factor:',random_factor,
                #'','boltz  factor:',boltzmann_factor,
                #'','REJECTED')
                                    
        self.save_trajectory()
        self.nsteps += 1
        
    def retry(self, prop_StrandA, prop_StrandB, catch = False):
        '''
        Finds DNA strands that fit the requirements using a while loop
        REQUIREMENTS:
            Excluded volume interactions
            Inside simulation box
            Intact [NOTE: no longer need to check for, with spherical angle propagation]
            & if catch, total pair count number increase
        '''
        # find valid configuration, need to wait for entire strand to change before excvol and inbox can fairly be applied
        if not catch:
            while not prop_StrandA.check_inbox(self.boxlims) or not prop_StrandB.check_inbox(self.boxlims): 
               prop_StrandA = self.StrandA.propose_change_whole()
               prop_StrandB = self.StrandB.propose_change_whole()
        if catch:
            while not prop_StrandA.check_inbox(self.boxlims) or not prop_StrandB.check_inbox(self.boxlims) or not self.StrandA.check_count_increase(prop_StrandA):
               prop_StrandA = self.StrandA.propose_change_whole()
               prop_StrandB = self.StrandB.propose_change_whole()
        return prop_StrandA, prop_StrandB
     
    # for data analysis
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
    
    def save_trajectory(self):
        
        new_trajA = []
        for seg in self.StrandA.dnastr:
            new_trajA.append(np.array([seg.position.x,seg.position.y,seg.position.z]))
        self.trajectoryA.append(new_trajA)
        
        new_trajB = []
        for seg in self.StrandB.dnastr:
            new_trajB.append(np.array([seg.position.x,seg.position.y,seg.position.z]))
        self.trajectoryB.append(new_trajB)
        
        self.eng_traj.append(self.energy)
        
        #self.centremass.append([self.StrandA.find_centremass(),self.StrandB.find_centremass()])
        
        totpair, selfpair, n_islands, av_R_islands, av_L_islands, av_sep_islands = self.islands_data()
        self.pair_counts.append([totpair, selfpair])
        self.n_islands.append(n_islands)
        self.av_R_islands.append(av_R_islands)
        self.av_L_islands.append(av_L_islands)
        self.av_sep_islands.append(av_sep_islands)
        
        if self.StrandA.interactions != [[[], [], []]]:
            self.interactions_traj.append(self.StrandA.interactions)
