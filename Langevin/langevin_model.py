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
        a truncated LJ potential for excluded volume effects
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
from scipy import signal
from typing import Tuple
from scipy import special
from scipy.constants import epsilon_0, Boltzmann

# # # UNITS # # #
kb = 1 #1.38e-23
temp = 310.15

# # # PARAMETERS # # #
# Worm Like Chain Bending
lp = 4.5 # persistence length, in coherence length diameter grains of 100 Angstroms
kappab = lp * kb * temp # bending stiffness
s = 0.4 # standard distance through chain separated by one Grain
k_bend = kappab/s # Bending stiffness constant

k_spring = 300*kb  # Spring constant for bonds

# Simulation Interaction Parameters
R_cut = 0.75 # cut off distance for electrostatic interactions, SURFACE to SURFACE distance, in helical coherence lengths (100 Angstroms)
self_interaction_limit = 5 # avoid interactions between grains in same strand

# Langevin 
lamb = 1 # damping coefficient
dt = 0.0005 # timestep



# # # Aux functions # # #
def gen_grains(coherence_lengths, start_position):
    strand = [Grain(start_position, np.zeros(3) )]
    for i in range(5*coherence_lengths):
        strand.append( Grain( strand[-1].position + np.array([0, 0.2, 0]), np.zeros(3) ) )
    return strand



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

    def __init__(self, homol=False):
        self.homol = homol     
        
    def kappa(self, n):
        return np.sqrt( 1/self.debye**2  +  n**2 * (2*np.pi / self.H)**2 ) # when n=0 kappa = 1/debye
        
    def f(self, n):
        return self.f1 * self.theta  +  self.f2 * (-1)**n * self.theta  -  (1 - self.f3*self.theta) * np.cos(0.4 * n * np.pi)    

    def nu(self, n, L):
        x = n**2 * L / self.lamb_c
        return ( 1 - np.exp(-x) ) / x 

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
        
        self.coeffs = 16 * np.pi**2 * self.sigma**2 / self.eps
        a0_coeff = 1/2
        a0_term1 = (1-self.theta)**2 * special.kn(0, Rrange/self.debye ) / ( (1/self.debye**2) * special.kn(1, self.r/self.debye )**2 )
        a0_term2 = 0
        for j in range(-1,2):
            for n in range(-1,2):
                a0_term2 += self.f(n)**2 / self.kappa(n)**2  *  special.kn(n-j, self.kappa(n)*Rrange)**2 * special.ivp(j, self.kappa(n)*self.r)  /  (  special.kvp(n, self.kappa(n)*self.r)**2 * special.kvp(j, self.kappa(n)*self.r)  ) 
        self.a0 = a0_coeff * (a0_term1 - a0_term2)
        self.a1 = self.f(1)**2 / self.kappa(1)**2 * special.kn(0, self.kappa(1)*Rrange ) / special.kvp(1, self.kappa(1)*self.r )**2
        self.a2 = self.f(2)**2 / self.kappa(2)**2 * special.kn(0, self.kappa(2)*Rrange ) / special.kvp(2, self.kappa(2)*self.r )**2

        if not self.homol:
            self.Eint = self.coeffs * ( self.a0  -  self.nu(1, Lrange) * self.a1 * np.cos(np.arccos(self.a1/(4*self.a2)))  +  self.nu(2, Lrange) * self.a2 * np.cos(2*np.arccos(self.a1/(4*self.a2))) ) * Lrange
        elif self.homol:
            self.Eint = self.coeffs * ( self.a0  -  self.a1*np.cos(np.arccos(self.a1/(4*self.a2)))  +  self.a2*np.cos(2*np.arccos(self.a1/(4*self.a2))) ) * Lrange
        
        self.Eint *= (4*np.pi*epsilon_0)
        self.Eint /= (Boltzmann * 300) 
        self.Eint *= 10**8 # from gaussian to in kbT units

    def find_energy(self, Lindex: int, R: float):
        '''Finds energy of ONE L, R point'''
        
        Lmin = 20  * 10**-10
        Lmax = 100000 * 10**-10
        Lstep = 20 * 10**-10 # grain diameter
        Lrange = np.linspace( Lmin, Lmax, int((Lmax-Lmin)/(Lstep))+1 )
        L = Lrange[Lindex-1]
        
        coeffs = 16 * np.pi**2 * self.sigma**2 / self.eps
        a0_coeff = 1/2
        a0_term1 = (1-self.theta)**2 * special.kn(0, R/self.debye ) / ( (1/self.debye**2) * special.kn(1, self.r/self.debye )**2 )
        a0_term2 = 0
        for j in range(-1,2):
            for n in range(-1,2):
                a0_term2 += self.f(n)**2 / self.kappa(n)**2  *  special.kn(n-j, self.kappa(n)*R)**2 * special.ivp(j, self.kappa(n)*self.r)  /  (  special.kvp(n, self.kappa(n)*self.r)**2 * special.kvp(j, self.kappa(n)*self.r)  ) 
        a0 = a0_coeff * (a0_term1 - a0_term2)
        a1 = self.f(1)**2 / self.kappa(1)**2 * special.kn(0, self.kappa(1)*R ) / special.kvp(1, self.kappa(1)*self.r )**2
        a2 = self.f(2)**2 / self.kappa(2)**2 * special.kn(0, self.kappa(2)*R ) / special.kvp(2, self.kappa(2)*self.r )**2

        if not self.homol:
            Eint = coeffs * ( a0  -  self.nu(1, L) * a1 * np.cos(np.arccos(a1/(4*a2)))  +  self.nu(2, L) * a2 * np.cos(2*np.arccos(a1/(4*a2))) ) * L
        elif self.homol:
            Eint = coeffs * ( a0  -  a1*np.cos(np.arccos(a1/(4*a2)))  +  a2*np.cos(2*np.arccos(a1/(4*a2))) ) * L
        
        Eint *= (4*np.pi*epsilon_0)
        Eint /= (Boltzmann * 300) 
        Eint *= 10**8 # from gaussian to in kbT units
        
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
        h = 0.0001 * 10**-10 # for differentiation by first principles
        dEdR = ( self.find_energy(Lindex, R+h) - self.find_energy(Lindex, R) ) / h
        
        return -1*dEdR
        
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
elstats = Electrostatics(homol=False)



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
        
        CHANGE: loop through a COMBINATIONS loop to speed up
        '''
        self.interactions = [ [[],[]] ] # starting with one empty 'island'
        # loop through all combinations
        for i, j in combinations(range(len(self.dnastr+other.dnastr)),2):
            if abs(i-j) <= self_interaction_limit: #and [self.num_segments-1, self.num_segments] not in [ [i,j] , [j,i] ]:
                continue # skips particular i, j if a close self interaction, avoiding application across strands
            igrain = self.dnastr[i] if i<self.num_segments else other.dnastr[i-self.num_segments] 
            jgrain = self.dnastr[j] if j<self.num_segments else other.dnastr[j-self.num_segments] # use correct strand
            R = jgrain.position - igrain.position
            R_norm = np.linalg.norm(R) - 0.2 # get surface - surface distance
            if R_norm < R_cut and R_norm != 0: # update dists attribute
                # find correct identities
                idnti = 's'+str(i) if i < self.num_segments else 'o'+str(i-self.num_segments)
                idntj = 's'+str(j) if j < self.num_segments else 'o'+str(j-self.num_segments)
                # add to interactions
                # check if add to any existing 'islands'
                island = self.find_island(idnti, idntj)
                if island != 'new':
                    self.interactions[island][0].append(idnti+' '+idntj)
                    self.interactions[island][1].append(R_norm)
                else:
                    self.interactions.append([[],[]]) # create new island
                    self.interactions[-1]    [0].append(idnti+' '+idntj)
                    self.interactions[-1]    [1].append(R_norm)
                
    def find_island(self, idnti, idntj):
        # check if add to any existing 'islands'
        # create list of possible configurations for an 'island'
        check_island_configurations = []
        for stepi in range(-1,2): # can change this for larger loops until interaction 'reset'
            for stepj in range(-1,2):
                check_island_configurations += [idnti[0]+str( int(idnti[1:])+stepi ) + ' ' +  idntj[0]+str( int(idntj[1:])+stepj )]
        # check possible configurations against existing islands
        for n in range(len(self.interactions)):
            for check_idnt in check_island_configurations:
                if check_idnt in self.interactions[n][0]:
                    return n
        return 'new'
                
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
                isle.append(Llist)
            else:
                isle.append([])
            
    def f_elstat(self, other, homol=False):
        ''' '''
        self.find_interactions(other)
        self.assign_L(other)

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
        return energy
        
    def eng_elastic(self) -> float:
        '''
        Energy term for bending of DNA strand from straight
        Uses small angle approx so finer coarse graining more accurate
        '''
        energy = 0
        for seg_index in range(1,self.num_segments-1):
            angle = self.find_angle(seg_index)
            energy += k_bend *  angle**2
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
            random_force = np.random.normal(0, np.sqrt(2 * lamb * kb * temp / dt), size=3)
            # Damping force
            grain.update_velocity(random_force,dt)
            
    def apply_box(self):
        for grain in self.StrandA.dnastr + self.StrandB.dnastr:
            for i in range(3):
                grain.velocity[i]*=-1 if abs(grain.position[0]+grain.radius>=self.boxlims[i]) and grain.position[i]*grain.velocity[i]>0 else 1

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
        
        totpair, selfpair = self.count_tot()
        self.pair_counts.append([totpair, selfpair])
        
        self.endtoends.append(self.endtoend(-1))
        
        #self.centremass.append([self.StrandA.find_centremass(),self.StrandB.find_centremass()])
    
    def endtoend(self, tindex):
        endtoendA = np.linalg.norm(self.trajectoryA[tindex][0] - self.trajectoryA[tindex][-1])
        endtoendB = np.linalg.norm(self.trajectoryB[tindex][0] - self.trajectoryB[tindex][-1])
        return endtoendA, endtoendB
    
    def count_tot(self):
        '''Discounts immediate neighbours from pairs
        Must be run only after gen_interactivity'''
        #comparearray = np.zeros(len(self.StrandA.interactivity))
        #pairsA = np.sum(np.array(self.StrandA.interactivity)!=comparearray)
        #pairsB = np.sum(np.array(self.StrandB.interactivity)!=comparearray)
        #totpairs = int((pairsA+pairsB)/2)
        count_total = 0
        count_pairs = 0
        for isle in self.StrandA.interactions:
            count_total += len(isle[0])
            for i in range(len(isle[0])):
                count_pairs += 1 if isle[0][i][0]==isle[0][i][4] else 0
        return count_total, count_pairs  
          
    def find_energy(self):
        '''Includes electrostatic and WLC bending energies ONLY'''
        return self.StrandA.eng_elastic() + self.StrandB.eng_elastic() + self.StrandA.eng_elstat(self.StrandB)
    
