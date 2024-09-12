# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 18:11:06 2024

@author: Ralph Holden

MODEL:
    polymer bead model - Worm-like-Chain, including excluded volume and specialised dsDNA interactions
    beads correspond to 1/5 correlation length as described in Kornyshev-Leiken theory
        as such, for non homologous DNA, only one consecutive helical coherence length can have attractive charged double helix interactions
   NOTE: edge effects not accounted for
    
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
"""
# imports
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from scipy import special
from scipy.constants import epsilon_0, Boltzmann

import sys
##import os

##script_dir = os.path.dirname(os.path.abspath(__file__))
##target_dir = os.path.join(script_dir, '../Electrostatics_functions/')
##sys.path.insert(0, os.path.abspath(target_dir))
##from Electrostatics_classholder import Electrostatics


# # # UNITS # # #
kb = 1 # 1.38e-23
temp = 310.15

# # # PARAMETERS # # #
# Worm Like Chain Bending
lp = 5 # persistence length, in coherence length diameter grains of 100 Angstroms
kappab = lp * kb * 300 # bending stiffness
s = 0.4 # standard distance through chain between three grains
k_bend = kappab/(2*s) # Bending stiffness constant

# Spring constant for bonds
k_spring = 30000000*kb

# Simulation Interaction Parameters
R_cut = 0.8 # cut off distance for electrostatic interactions, SURFACE to SURFACE distance, in helical coherence lengths (100 Angstroms) therefore 7.5 nm in real units
self_interaction_limit = 6 # avoid interactions between grains in same strand
wall_dist = 0.2
##homology_set = True # False -> NON homologous

# Langevin 
dt = 0.0001 # timestep, per unit mass 
dynamic_coefficient_friction = 0.00069130 # in Pa*s, for water at 310.15 K & 1 atm, from NIST
l_kuhn = lp # persistence length
slender_body = 4*np.pi * dynamic_coefficient_friction * l_kuhn / np.log( l_kuhn / 0.2 ) # damping coefficient - perpendicular motion case from Slender Body Theory of Stokes flow
gamma = 0.5 # or slender body

correlation_length = 5 # number of grains with (fully) correlated fluctuations
grain_mass = 1
grain_radius = 0.1 # grain radius

# define parameters for Langevin modified Velocity-Verlet algorithm - M. Kroger
xi = 2/dt * gamma # used instead of gamma for Langevin modified Velocity-Verlet
half_dt = dt/2
applied_friction_coeff = (2 - xi*dt)/(2 + xi*dt)
fluctuation_size = np.sqrt( grain_mass * kb * temp * xi * half_dt ) # takes dt into account. should it be /grain_mass ?
rescaled_position_step = 2*dt / (2 + xi*dt)

no_fluctuations = False # allows testing for minimum internal energy


# # # Aux functions # # #

# # # ELECTROSTATICS # # #
# next improvement: take repulsions from theory, rather than LJ potentials
# next improvement: interpolation between homologous and non homologous interaction

def interaction_nonhomologous(R_norm, doforce=True):
    '''ASSUMING all non homologous interactions are REPULSIVE ONLY, on the basis that non homologous strands are unlikely to converge'''
    # LJ params
    eps = 300/4*kb # 1/4 of interaction strength ~1 kbT
    sig = grain_radius*2 # particle diameter
    if not doforce: # energy
        return eps * (sig/R_norm)**12
    elif doforce: # force magnitude
        return -12*eps * sig**12/R_norm**13

def interaction_homologous(R_norm, doforce=True):
    '''For grain with matching dnastr index ONLY (for now)'''
    # LJ params
    eps = 300/4*kb # 1/4 of interaction strength ~1 kbT
    sig = grain_radius*2 # particle diameter
    if not doforce: # energy
        return eps * ( (sig/R_norm)**12 - (sig/R_norm)**6 )
    elif doforce: # force magnitude
        return eps * ( -12*(sig**12/R_norm**13) + 6*(sig**6/R_norm**7) )

class Start_position:
    '''
    Initialises the starting positions of a DNA strand
    
    INPUTS:
        num_segments: int
        xstart      : float, starting position of first DNA 'grain'
        ystart      : float
        zstart      :
    
    METHODS:
        create_strand_curved()  : initalises the positions around semi-circles radius l_p/pi
                                  NOTE: causes start in non equilibrium bond distances
        create_strand_straight(): initalises the positions in a simple straight line
                                  NOTE: causes start in non equilibrium curvature
        plot_start()            
    '''
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



# Pairing interaction energy
##elstats = Electrostatics(homol = homology_set)



class Grain():
    def __init__(self, position: np.array, velocity: np.array, radius = 0.1):
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.ext_force = np.zeros(3, dtype=np.float64)
        self.radius = radius

    def update_position(self, timestep):
        # Update position based on the velocity
        # from Simulation.run_step() -> timestep = rescaled_position_step
        self.position += self.velocity * timestep
        
    def update_force(self, ext_force):
        self.ext_force += ext_force
        
    def copy(self):
        return Grain(self.position, self.velocity, self.radius)
    
    
    
class Strand:
    
    def __init__(self, grains):
        self.dnastr = grains
        self.num_segments = len(grains)
        
        # data and code efficiency
        self.interactions = [],[] # list[0] is of R_norm, list[1] is of homologous (True) vs nonhomologous (False) 
        self.angle_list = []
        
        # Gives list indexes that can break the self interaction lower limit, by interacting ACROSS the strands
        self.cross_list = [] # defined here as to avoid repetitions. 
        for i in range(self.num_segments - self_interaction_limit, self.num_segments):
            for j in range(i, i+self_interaction_limit+1):
                if j >= self.num_segments:
                    self.cross_list.append([i,j]) 
        
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
            self.dnastr[i].update_force(force)
            self.dnastr[i+1].update_force(-force)
        
    def f_wlc(self):
        
        # reset angle_list attribute
        self.angle_list = []
        
        for i in range(1, self.num_segments - 1):
            # Vectors between adjacent grains
            r1 = self.dnastr[i-1].position - self.dnastr[i].position
            r2 = self.dnastr[i+1].position - self.dnastr[i].position
            r1_norm = np.linalg.norm(r1)
            r2_norm = np.linalg.norm(r2)
            # Cosine of the angle between r1 and r2
            cos_theta = np.dot(r1, r2) / (r1_norm * r2_norm)
            theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            
            # save angles for calculating elastic energy
            self.angle_list.append( theta - np.pi )
            
            if np.isclose(theta,np.pi):
                continue # angle equivalent to 0 therefore no torque required
                
            force_magnitude = -k_bend * (theta - np.pi) 
            force_direction = r1+r2
            force_direction /= np.linalg.norm(force_direction) if np.linalg.norm(force_direction) != 0 else 1
            force = force_magnitude * force_direction
            self.dnastr[i].update_force(force)
    
    # electrostatic interaction        
    def f_elstat(self, other):
        ''' 
        Takes ALL interactions below the cutoff distance. Only needs calling once per half-step.
        '''
        # reset interactions attribute
        self.interactions = [],[]
        
        for i,j in combinations(range(self.num_segments+other.num_segments), 2):
            
            if abs(i-j) <= self_interaction_limit and [i,j] not in self.cross_list:
                continue # skips particular i, j if a close self interaction, avoiding application across strands
            
            # assign grains
            igrain = self.dnastr[i] if i<self.num_segments else other.dnastr[i-self.num_segments] 
            jgrain = self.dnastr[j] if j<self.num_segments else other.dnastr[j-self.num_segments] # use correct strand
            
            # homology of interaction
            ishomol = True if abs(i-j)==self.num_segments else False
            
            # find intergrain distance
            R = jgrain.position - igrain.position
            R_norm = np.linalg.norm(jgrain.position - igrain.position)
            R /= R_norm
            
            # code efficiency 
            if R_norm > R_cut:
                continue # skip this interaction
            
            # save to self.interactions, for efficiency of calculating electrostatic energy
            # done before rescaling R_norm
            self.interactions[0].append(R_norm) , self.interactions[1].append(ishomol)
            
            # code safety, rescale R_norm to avoid dangerous conditions
            R_norm = wall_dist if R_norm < wall_dist else R_norm 
            
            # linear interaction force
            f_lin = interaction_homologous(R_norm, doforce=True) if ishomol else interaction_nonhomologous(R_norm, doforce=False)
            
            # apply force
            igrain.update_force(+1*f_lin*R)
            jgrain.update_force(-1*f_lin*R)
        
    # for energies, not including bond springs or translational energy
    def eng_elstat(self, other, homol=False):
        '''
        Does electrostatic energy of BOTH strands
        In each step, use after f_elstat() so gen functions do not have to be repeated, homol argument in f_elstat() defines homologicity
        '''
        energy = 0.0
        for p in range(len(self.interactions[0])):
            R_norm = self.interactions[0][p]
            ishomol = self.interactions[1][p]
            energy += interaction_homologous(R_norm, doforce=False) if ishomol else interaction_nonhomologous(R_norm, doforce=False)
        return energy / (kb*300) # give energy in kbT units
        
    def eng_elastic(self) -> float:
        '''
        Energy term for bending of DNA strand from straight
        Uses small angle approx so finer coarse graining more accurate
        '''
        energy = 0
        for angle in self.angle_list:
            energy += 1/2 * k_bend *  angle**2
        return energy / (kb*300) # give energy in kbT units
    
    
    
class Simulation:
    def __init__(self, StrandA: Strand, StrandB: Strand, boxlims: np.array):
        self.StrandA = StrandA
        self.StrandB = StrandB
        
        self.boxlims = boxlims
    
        # data
        self.trajectoryA = []
        self.trajectoryB = []
        self.energy_traj = []
        self.endtoend_traj = []
        self.mean_curvature_traj = []
        self.std_curvature_traj = []
        self.total_pairs_traj = []
        self.homol_pairs_traj = []
        self.homol_pair_dist_traj = []
        
        #self.record()

    # for time integration / evolution
    def run_step(self):
        '''
        Langevin Modified Velocity-Verlet Algorithm
        Taken from Models for polymeric and anisotropic liquids, M. Kröger, 2005
        '''
        # reset and calculate external forces, information stored in Grain attribute
        self.calc_external_force()
        
        # take random fluctuation length for each 'correlation length'
        fluctuation_list = self.langevin_fluctuations()
        
        self.update_velocities_first_halfstep( fluctuation_list )
        
        # apply position update
        self.update_positions()
        
        # reset and calculate external forces
        self.calc_external_force()
        
        self.update_velocities_second_halfstep( fluctuation_list )
        
        # save data
        self.record()
        
    def calc_external_force(self):
        '''
        Reset and calculate the external force acting on each grain
        '''
        # reset Grain attribute ext_force 
        for g in self.StrandA.dnastr + self.StrandB.dnastr:
            g.ext_force = np.zeros(3, dtype=np.float64)
        # find external force from chain bond, bending, electrostatics and boundary conditions
        self.StrandA.f_bond(), self.StrandB.f_bond()
        self.StrandA.f_wlc(), self.StrandB.f_wlc() 
        self.StrandA.f_elstat(self.StrandB)
        self.apply_box()
        
    def langevin_fluctuations(self):
        '''
        Build list of fluctuations for each grain. Correlated within each 'correlation length'.
        Correlation length usually kuhn length (25) or, less often, helical coherence length (5)
        
        Applies UNcorrellated brownian force to each 'correlation length'
        Fully correllated within fluctuation 'correlation length'
        '''
        # if no fluctuations specified (for bug testing), return array of zeroes
        if no_fluctuations:
            return np.zeros(self.StrandA.num_segments+self.StrandB.num_segments)
        
        # reset lists
        fluctuationA , fluctuationB = [] , []
        
        # build lists, may have different lengths
        scaling_factor = 1 #e-8
        for i in range(int(np.ceil(self.StrandA.num_segments/correlation_length))):
            fluctuationA += [(np.random.normal(0, fluctuation_size, size=1) *scaling_factor )[0]] * correlation_length
        for i in range(int(np.ceil(self.StrandB.num_segments/correlation_length))):
            fluctuationB += [(np.random.normal(0, fluctuation_size, size=1) *scaling_factor )[0]] * correlation_length
        
        # correct length
        fluctuationA, fluctuationB = fluctuationA[:self.StrandA.num_segments] , fluctuationB[:self.StrandB.num_segments] 
        
        # put together, for form taken by functions
        fluctuation_list = fluctuationA + fluctuationB
        
        return fluctuation_list
        
    def update_velocities_first_halfstep(self, fluctuation_list):
        for i, grain in enumerate(self.StrandA.dnastr + self.StrandB.dnastr):
            grain.velocity += half_dt * grain.ext_force / grain_mass # apply external force
            grain.velocity += fluctuation_list[i] # apply fluctuation 
    
    def update_positions(self):
        for grain in self.StrandA.dnastr + self.StrandB.dnastr:
            grain.update_position( rescaled_position_step )     
            
    def update_velocities_second_halfstep(self, fluctuation_list):
        for i, grain in enumerate(self.StrandA.dnastr + self.StrandB.dnastr):
            grain.velocity *= applied_friction_coeff # apply friction
            grain.velocity += half_dt * grain.ext_force / grain_mass # apply external force
            grain.velocity += fluctuation_list[i] # apply fluctuation 
            
    def apply_box(self):
        '''
        Constant force be applied to entire strand when one part strays beyond box limits
        
        Update required: different boundary conditions
            - soft walls (weak spring)
            - osmotic pressure (constant central force)
        '''
        returning_force_mag = 10000
        for strand in self.StrandA,self.StrandB:
            returning_force = np.array([0., 0., 0.])
            for i in range(3):
                for grain in strand.dnastr:
                    if grain.position[i] + grain.radius > self.boxlims[i]:
                        returning_force[i] += -1*returning_force_mag
                        continue # no need to check for any more
                    if grain.position[i] - grain.radius < -self.boxlims[i]:
                        returning_force[i] += +1*returning_force_mag
                        continue
            if not np.isclose(np.linalg.norm(returning_force), 0):
                    for grain in strand.dnastr:
                        grain.update_force(returning_force)
      
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
        
        self.energy_traj.append(self.find_energy())
        
        self.endtoend_traj.append(self.find_endtoend(-1))
        
        mean_curvature, std_curvature = self.find_curvature()
        self.mean_curvature_traj.append(mean_curvature)
        self.std_curvature_traj.append(std_curvature)
        
        total_pairs, homol_pairs, homol_pair_dist = self.find_pair_data()
        self.total_pairs_traj.append(total_pairs)
        self.homol_pairs_traj.append(homol_pairs)
        self.homol_pair_dist_traj.append(homol_pair_dist)
        
    def find_curvature(self):
        return np.mean( self.StrandA.angle_list + self.StrandB.angle_list ) , np.std( self.StrandA.angle_list + self.StrandB.angle_list )
    
    def find_pair_data(self):
        total_pairs = len(self.StrandA.interactions[0])
        homol_pairs = np.sum(self.StrandA.interactions[1])
        homol_R_norm_list = np.delete(self.StrandA.interactions[0],np.array(self.StrandA.interactions[1])==False)
        homol_pair_dist = np.mean(homol_R_norm_list) if len(homol_R_norm_list)>0 else None
        return total_pairs, homol_pairs, homol_pair_dist
    
    def find_endtoend(self, tindex):
        endtoendA = np.linalg.norm(self.trajectoryA[tindex][0] - self.trajectoryA[tindex][-1]) + 0.2
        endtoendB = np.linalg.norm(self.trajectoryB[tindex][0] - self.trajectoryB[tindex][-1]) + 0.2 # account for size of particle
        return endtoendA, endtoendB
          
    def find_energy(self):
        '''Includes electrostatic and WLC bending energies ONLY'''
        return self.StrandA.eng_elastic() + self.StrandB.eng_elastic() + self.StrandA.eng_elstat(self.StrandB)
    