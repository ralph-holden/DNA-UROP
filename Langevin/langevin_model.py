# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 18:11:06 2024

@author: Ralph Holden

MODEL:
    polymer bead-chain model, WLC including excluded volume and specialised dsDNA interactions
    beads correspond to 1/25 a persistence length, or 1/5 correlation length as described in Kornyshev-Leiken theory
        for non-homologous DNA, only one consecutive helical coherence length can have attractive charged double helix interactions
   NOTE: edge effects not accounted for
    
CODE & SIMULATION:
    Langevin dynamics used to propagate each grain in the dsDNA strands
    Additional forces involved are; 
        harmonic grain springs for keeping the strand intact (artifical 'fix')
        harmonic angular springs for the Worm-Like Chain
        conditional electrostatic interactions as described in Kornyshev-Leikin theory
        + keeping inside the simulation box (confined DNA, closed simulation, artificicial 'fix' to avoid lost particles)
    Energy dependant on:
        worm like chain bending (small angle approx -> angular harmonic)
        conditional electrostatic interactions as described in Kornyshev-Leikin theory
"""
# # # Imports # # #
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from scipy import special
from scipy import signal
from scipy.constants import epsilon_0, Boltzmann

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(script_dir, '../Electrostatics_functions/')
sys.path.insert(0, os.path.abspath(target_dir))
from Electrostatics_classholder import Electrostatics


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
k_spring = 300000000*kb

# Simulation Interaction Parameters
R_cut = 0.4 # cut off distance for electrostatic interactions, SURFACE to SURFACE distance, in helical coherence lengths (100 Angstroms) therefore 7.5 nm in real units
self_interaction_limit = 6 # avoid interactions between grains in same strand
wall_dist = 0.2
homology_set = True # False -> NON homologous
pair_count_upper_dist = 0.25

# Langevin 
dt = 0.000025 # timestep, per unit mass 
dynamic_coefficient_friction = 0.00069130 # in Pa*s, for water at 310.15 K & 1 atm, from NIST
l_kuhn = lp # persistence length
slender_body = 4*np.pi * dynamic_coefficient_friction * l_kuhn / np.log( l_kuhn / 0.2 ) # damping coefficient - perpendicular motion case from Slender Body Theory of Stokes flow
gamma = 0.5 # or slender body

correlation_length = 5 # number of grains with (fully) correlated fluctuations (note: not coherence_lengths)
grain_mass = 1
grain_radius = 0.1 # grain radius

# define parameters for Langevin modified Velocity-Verlet algorithm - M. Kroger
xi = 2/dt * gamma # used instead of gamma for Langevin modified Velocity-Verlet
half_dt = dt/2
applied_friction_coeff = (2 - xi*dt)/(2 + xi*dt)
fluctuation_size = np.sqrt( grain_mass * kb * temp * xi * half_dt ) # takes dt into account. should it be /grain_mass ?
rescaled_position_step = 2*dt / (2 + xi*dt)

no_fluctuations = False # if True, allows testing for minimum internal energy

# boundary conditions
# settings
osmotic_pressure_set = False
soft_vesicle_set = False # if both set False, 'sharp_return' style is used
# boundary force sizes
osmotic_pressure_constant = 1000
soft_vesicle_k = 100
returning_force_mag = 1000
# Container limits, for soft_vesicle and sharp_return settings
container_size = 15


# # # Aux functions # # #

# # # ELECTROSTATICS # # #
elstats = Electrostatics(homology_set)

# Initial configuration
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
        self.isinteract = np.zeros(self.num_segments)==np.ones(self.num_segments)
        self.interactions = []
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
            
    def apply_distance_constraints(self):
        for i in range(self.num_segments-1):
            grain1 = self.dnastr[i]
            grain2 = self.dnastr[i+1]
            
            p1 = grain1.position
            p2 = grain2.position
            inter_vector = p2 - p1
            
            distance = np.linalg.norm( inter_vector )
            required_distance = grain1.radius+grain2.radius
            
            if np.isclose(distance, required_distance):
                continue # does not need adjustment
            
            # adjust all others
            for j in range(i+1,self.num_segments):
                self.dnastr[j].position += inter_vector * (required_distance - distance)
                
    # WLC bending energies
    def angle_derivative(self, x_p1, x_m, x_m1,    y_p1, y_m, y_m1,    z_p1, z_m, z_m1):
        
        # Common terms used in multiple places
        term_x_m1_m = (x_m1 - x_m)**2 + (y_m1 - y_m)**2 + (z_m1 - z_m)**2
        term_x_m_p1 = (x_m - x_p1)**2 + (y_m - y_p1)**2 + (z_m - z_p1)**2
        term_cross_product = (x_m1 - x_m) * (x_m - x_p1) + (y_m1 - y_m) * (y_m - y_p1) + (z_m1 - z_m) * (z_m - z_p1)
    
        # Numerator terms
        term1 = (-x_m1 + 2*x_m - x_p1) * term_x_m1_m * term_x_m_p1
        term2 = (-x_m + x_p1) * term_x_m1_m * term_cross_product
        term3 = (x_m1 - x_m) * term_x_m_p1 * term_cross_product
    
        numerator = term1 + term2 + term3
    
        # Denominator terms
        denominator = (term_x_m1_m**(3/2)) * (term_x_m_p1**(3/2))
    
        # Final expression
        result = numerator / denominator
        return result

    def f_wlc_new(self):

        # extract positions for grain attributes
        pos_array = []
        for i in range(self.num_segments):
            pos_array.append(self.dnastr[i].position)
        pos_array = np.array(pos_array)
            
        # all middle strand forces, sum of m-1 and m+1 contributions
        x_m     , y_m     , z_m      = pos_array[1:-1,0], pos_array[1:-1,1], pos_array[1:-1,2] # all middle strands
        x_minus1, y_minus1, z_minus1 = pos_array[:-2,0] , pos_array[:-2,1] , pos_array[:-2,2]  # m-1 array
        x_plus1 , y_plus1 , z_plus1  = pos_array[2:,0]  , pos_array[2:,1]  , pos_array[2:,2]   # m+1 array
        
        # derivatives for x
        xforces = -k_bend * self.angle_derivative(x_plus1, x_m, x_minus1,   y_plus1, y_m, y_minus1,   z_plus1, z_m, z_minus1)
        
        # derivatives for y
        yforces = -k_bend * self.angle_derivative(y_plus1, y_m, y_minus1,   x_plus1, x_m, x_minus1,   z_plus1, z_m, z_minus1)
        
        # derivatives for z
        zforces = -k_bend * self.angle_derivative(z_plus1, z_m, z_minus1,   x_plus1, x_m, x_minus1,   y_plus1, y_m, y_minus1)
        
        # apply forces in grain attributes
        # if applied to single grain only
        for i in range(1,self.num_segments-1):
                self.dnastr[i].update_force(np.array([ xforces[i-1], yforces[i-1], zforces[i-1]]))
        
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
        
    def find_angle(self, seg_index):
        p1 = self.dnastr[seg_index-1].position
        p2 = self.dnastr[seg_index].position
        p3 = self.dnastr[seg_index+1].position
        
        # colinear vectors
        if np.isclose(p3-p2,p2-p1).all():
            return 0
        # return 180 - angle
        return np.pi - np.arccos(np.dot(p3-p2,p1-p2) / (np.linalg.norm(p3-p2)*np.linalg.norm(p1-p2) ) )
    
    # electrostatic interaction
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
            R_norm = np.linalg.norm(R) # get interaxial distance
            
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
        '''
        IN / OUT ISLAND REQUIREMENT  ***EXPERIMETNAL*** in other file
        if within the same island, if either idnti OR idntj already there, cut one
        but if in different islands, cut one ONLY if idnti AND idntj are there
        
        R REQUIREMENT
        if already there, and the interaction is closer, then ignore the new interaction
        however, if already there but the old interaction is further, remove the previous interaction, and allow the new one
        '''
        ignore_interaction = False # default, unless proven otherwise
        
        if type(island) == int:
            # within same island
            idnt_match_index = []
            for i in range(len( self.interactions[island][0] )):
                if idnti + ' ' + idntj == self.interactions[island][0][i]:
                    continue
                idnt_match_index += [i] if idnti in self.interactions[island][0][i].split(' ') else []
                idnt_match_index += [i] if idntj in self.interactions[island][0][i].split(' ') else []
            for i in idnt_match_index:
                R_compare = self.interactions[island][1][i]
                if R_compare < R_norm or np.isclose(R_norm,R_compare):
                    ignore_interaction = True
                    return ignore_interaction
                elif R_compare > R_norm:
                    # do not ignore, remove previous interaction
                    self.interactions[island][0].remove(self.interactions[island][0][i])
                    self.interactions[island][1].remove(self.interactions[island][1][i])
                    for i in range(len(idnt_match_index)):
                        idnt_match_index[i] -= 1
        return ignore_interaction
                   
    def assign_L(self, other):
        '''
        Gives 'L' length of interacting dsDNA in 'island' of interaction
        Prioritises closest interacting grain to centre L at L=1, from here, interaction length builds up within island
        '''
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
                    for index, inter in enumerate(isle[0]):
                        IDi_n = int(inter.split(' ')[0][1:])
                        IDj_n = int(inter.split(' ')[1][1:])
                        if abs(IDi_n - IDj_n) <= 15: # assume recognition funnel interaction zero after 3 lc
                            Llist[index] = 'homol ' + str(abs(IDi_n - IDj_n))
                isle.append(Llist)
            else:
                isle.append([])
                  
    def f_elstat(self, other):
        ''' 
        Calculates force from electrostatic interaction (homologous and NON) on each interacting grain
        '''
        self.find_interactions(other)
        self.assign_L(other)
        
        # must only take MOST SIGNIFICANT (closest) interaction for each grain in an interacting 'island'

        for isle in self.interactions:
            for i in range(len(isle[0])):
                dist_walled = isle[1][i]*10**-8 if isle[1][i] >= wall_dist else 0.2e-8 # wall the repulsive potential
                felstat = elstats.force(isle[2][i], dist_walled) # change to metres
                # identify relevant grains
                idnt1 = isle[0][i].split(' ')[0]
                idnt2 = isle[0][i].split(' ')[1]
                grain1 = self.dnastr[ int(idnt1[1:]) ] if idnt1[0]=='s' else other.dnastr[ int(idnt1[1:]) ]
                grain2 = self.dnastr[ int(idnt2[1:]) ] if idnt2[0]=='s' else other.dnastr[ int(idnt2[1:]) ]
                fvec = grain2.position - grain1.position
                fvec /= np.linalg.norm(fvec) if np.linalg.norm(fvec) != 0 else 1
                grain1.update_force(+1*felstat*fvec)
                grain2.update_force(-1*felstat*fvec)
                
    def f_homology_recognition(self, other):
        k_recognition = 10000 # need to choose value
        if homology_set:
            for isle in self.interactions:
                frec_magnitude = 0.0
                for pi in range(len(isle[0])):
                    if isle[0][pi][0]!=isle[0][pi][4]:
                        gi_A, gi_B = int(isle[0][pi].split(' ')[0][1:]) , int(isle[0][pi].split(' ')[1][1:])
                        delta_AB = gi_A - gi_B
                        if abs( delta_AB ) <= 15 and delta_AB != 0:        
                            # find direction of force, if gi_A > gi_B, then direction towards lower A and higher B, vice versa
                            direction_str = True if delta_AB>0 else False # # # NOT ALWAYS TRUE # # #
                            # add magnitude to recognition force
                            frec_magnitude = k_recognition * np.exp(-delta_AB/(5*1)) * delta_AB * len(isle)
                            continue
                # apply total recognition force, to whole island
                if frec_magnitude != 0.0:
                    for pi in range(len(isle[0])):
                        # find grain
                        gi_A, gi_B = int(isle[0][pi].split(' ')[0][1:]) , int(isle[0][pi].split(' ')[1][1:])
                        # find direction down contour
                        if gi_A == self.num_segments-1 or gi_B == other.num_segments-1:
                            continue
                        direction_vec_A = self.dnastr[gi_A].position - self.dnastr[gi_A+1].position if direction_str else self.dnastr[gi_A].position - self.dnastr[gi_A-1].position
                        direction_vec_B = other.dnastr[gi_B].position - other.dnastr[gi_B+1].position if not direction_str else self.dnastr[gi_B].position - self.dnastr[gi_B-1].position
                        # apply force
                        self.dnastr[gi_A].update_force(direction_vec_A*frec_magnitude)
                        other.dnastr[gi_B].update_force(direction_vec_B*frec_magnitude)
        
    # for energies, not including bond springs or translational energy
    def eng_elstat(self, other):
        '''
        Does electrostatic energy of BOTH strands
        In each step, use after f_elstat() so gen functions do not have to be repeated, homol argument in f_elstat() defines homologicity
        '''
        energy = 0.0
        for isle in self.interactions:
            #if isle[1] == [] or isle[2] == []:
            #    continue
            for n in range(len( isle[1] )):
                g_R = isle[1][n]
                dist_walled = g_R*10**-8 if g_R >= wall_dist else 0.2e-8 # wall the repulsive potential
                g_L = isle[2][n]
                ishomol = type(g_L) == str
                if not ishomol:
                    energy += elstats.find_energy(g_L, dist_walled) - elstats.find_energy(g_L-1, g_R*10**-8) # remove 'built-up' energy over L w/ different R
                elif ishomol:
                    energy += elstats.find_energy(g_L, dist_walled) # energy is per unit length
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
        
        self.energy_traj = []
        
        self.endtoend_traj = []
        self.mean_curvature_traj = []
        self.std_curvature_traj = []
        
        self.homol_pairs_traj = []
        self.homol_pair_dist_traj = []
        self.terminal_dist_traj = []
        self.n_loops_traj = []
        
        self.total_pairs_traj = []
        self.R_islands_traj = []
        self.n_islands_traj = []
        self.L_islands_traj = [] 
        self.sep_islands_traj = []
        
        self.interactions_traj = []
        
        self.MSD_cm_traj = []
        #self.record()

    # for time integration / evolution
    def run_step(self, fluctuation_factor=1.0):
        '''
        Langevin Modified Velocity-Verlet Algorithm
        Taken from Models for polymeric and anisotropic liquids, M. Kröger, 2005
        '''
        # reset and calculate external forces, information stored in Grain attribute
        #self.calc_external_force()
        
        # take random fluctuation length for each 'correlation length'
        fluctuation_list = self.langevin_fluctuations(fluctuation_factor)
        
        self.update_velocities_first_halfstep( fluctuation_list )
        
        self.update_positions()
        #self.StrandA.apply_distance_constraints(), self.StrandB.apply_distance_constraints()
        
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
        self.StrandA.f_homology_recognition(self.StrandB)
        self.apply_box()
        
    def langevin_fluctuations(self, fluctuation_factor):
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
        adjusted_fluctuation_size = fluctuation_size * fluctuation_factor
        for i in range(int(np.ceil(self.StrandA.num_segments/correlation_length))):
            fluctuationA += [(np.random.normal(0, adjusted_fluctuation_size, size=3) )] * correlation_length
        for i in range(int(np.ceil(self.StrandB.num_segments/correlation_length))):
            fluctuationB += [(np.random.normal(0, adjusted_fluctuation_size, size=3) )] * correlation_length
        
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
        '''
        if osmotic_pressure_set: # independent of container_size
            # calculate vector from centre of mass towards centre of box (0, 0, 0)
            for strand in self.StrandA, self.StrandB:
                tot_gpos = []
                for g in strand.dnastr:
                    tot_gpos += [g.position]
                centre_mass_vec = np.mean(tot_gpos,axis=0)
                centre_mass_vec /= np.linalg.norm(centre_mass_vec)
                # apply osmotic pressure to all grains
                for g in strand.dnastr:
                    g.update_force( -osmotic_pressure_constant * centre_mass_vec )
        
        if soft_vesicle_set: # spherical, returning force to centre, only largest boxlim matters
            for strand in self.StrandA, self.StrandB:
                for g in strand.dnastr:
                    if np.linalg.norm(g.position) > container_size:
                        g.update_force( -soft_vesicle_k * (np.linalg.norm(g.position) - container_size) * g.position/np.linalg.norm(g.position) )
        
        if not osmotic_pressure_set and not soft_vesicle_set: # sharp return style
            for strand in self.StrandA,self.StrandB:
                
                centremass_position = np.zeros(3)
                for g in strand.dnastr:
                    centremass_position += g.position
                centremass_position /= strand.num_segments
                
                if np.linalg.norm(centremass_position) > container_size:
                    for g in strand.dnastr:
                        g.update_force( -returning_force_mag * centremass_position/np.linalg.norm(centremass_position) )
      
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
        
        '''
        self.MSD_cm_traj = []
        '''
        
        total_pairs, R_islands, n_islands, L_islands, sep_islands = self.find_island_data()
        self.total_pairs_traj.append(total_pairs)
        self.R_islands_traj.append(R_islands)
        self.n_islands_traj.append(n_islands)
        self.L_islands_traj.append(L_islands)
        self.sep_islands_traj.append(sep_islands)
        
        homol_pairs, homol_pair_dist, terminal_dist, n_loops = self.find_pair_data()
        self.homol_pairs_traj.append(homol_pairs)
        self.homol_pair_dist_traj.append(homol_pair_dist)
        self.terminal_dist_traj.append(terminal_dist)
        self.n_loops_traj.append(n_loops)
        
        if self.StrandA.interactions != [[[], [], []]]:
            self.interactions_traj.append(self.StrandA.interactions)
    
    def find_endtoend(self, tindex):
        endtoendA = np.linalg.norm(self.trajectoryA[tindex][0] - self.trajectoryA[tindex][-1]) + 0.2
        endtoendB = np.linalg.norm(self.trajectoryB[tindex][0] - self.trajectoryB[tindex][-1]) + 0.2 # account for size of particle
        return endtoendA, endtoendB
    
    def find_curvature(self):
        return np.mean( self.StrandA.angle_list + self.StrandB.angle_list ) , np.std( self.StrandA.angle_list + self.StrandB.angle_list )
    
    def find_MSD(self):
        pass
    
    def find_island_data(self):
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
            return 0, None, 0, None, None
        
        n_islands = len(self.StrandA.interactions) # number of loops / islands
        
        total_pairs = 0
        av_R_islands = 0
        start_end_g = [],[]
        
        for isle in self.StrandA.interactions:
            
            total_pairs += np.sum( np.array(isle[1]) < np.array([pair_count_upper_dist]) )
            
            av_R_islands += np.mean( isle[1] )
            
            for i in [0,1]:
                for j in [0,-1]:
                    start_end_g[i].append( int( isle[0][j] . split(' ')[i][1:] ) )
        
        av_R_islands /= n_islands
        
        L_islands    = np.diff( start_end_g ) [:,  ::2] + 1
        sep_islands  = np.diff( start_end_g ) [:, 1::2] - 1 if len( np.diff( start_end_g ) [:, 1::2] ) != 0 else None
        av_L_islands   = np.mean( L_islands   ) * 2*grain_radius
        av_sep_islands = np.mean( sep_islands ) * 2*grain_radius if sep_islands.any() != None else None # convert to units of coherence lengths
        
        return total_pairs, av_R_islands, n_islands, av_L_islands, av_sep_islands
    
    def find_pair_data(self):
        # find homologous pair count
        homol_R_norm_list = np.linalg.norm(np.array(self.trajectoryA[-1]) - np.array(self.trajectoryB[-1]), axis=1)
        homol_pairs = np.sum( homol_R_norm_list < np.array([pair_count_upper_dist]) ) # <0.25 are 'caught' in recognition well
        
        # separation of pairs
        homol_pair_dist = np.mean( homol_R_norm_list )
        terminal_dist = np.mean( [np.linalg.norm(self.StrandA.dnastr[0].position - self.StrandB.dnastr[0].position), np.linalg.norm(self.StrandA.dnastr[-1].position - self.StrandB.dnastr[-1].position)] )
        
        # number of 'loops'
        R_norm_list = []
        for i in range(len(self.StrandA.interactions)):
            R_norm_list += self.StrandA.interactions[i][1]
        R_norm_bool = np.array(R_norm_list) < pair_count_upper_dist
        n_loops = len( signal.find_peaks( np.concatenate( ( np.array([False]), R_norm_bool, np.array([False]) ) ) )[0] )
        
        return homol_pairs, homol_pair_dist, terminal_dist, n_loops
    
    def find_energy(self):
        '''Includes electrostatic and WLC bending energies ONLY'''
        return self.StrandA.eng_elastic() + self.StrandB.eng_elastic() + self.StrandA.eng_elstat(self.StrandB)
