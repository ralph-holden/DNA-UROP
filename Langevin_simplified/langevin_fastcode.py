# -*- coding: utf-8 -*-
"""
Created on Sun Sept 22 14:11:06 2024

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
# # # IMPORTS # # #
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from scipy import special
from scipy import signal
from scipy.constants import epsilon_0, Boltzmann
from numba import njit, float64
from numba.types import ListType
from numba.typed import List

import pickle
import sys
import os
import logging
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

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
##homology_set = True # False -> NON homologous

# Langevin 
dt = 0.000025 # timestep, per unit mass 
dynamic_coefficient_friction = 0.00069130 # in Pa*s, for water at 310.15 K & 1 atm, from NIST
l_kuhn = lp # persistence length
slender_body = 4*np.pi * dynamic_coefficient_friction * l_kuhn / np.log( l_kuhn / 0.2 ) # damping coefficient - perpendicular motion case from Slender Body Theory of Stokes flow
gamma = 0.5 # or slender body

correlation_length = 5 # number of grains with (fully) correlated fluctuations (note: not coherence_lengths)
grain_mass = 1.0
grain_radius = 0.1 # grain radius

# define parameters for Langevin modified Velocity-Verlet algorithm - M. Kroger
xi = 2/dt * gamma # used instead of gamma for Langevin modified Velocity-Verlet
half_dt = dt/2
applied_friction_coeff = (2 - xi*dt)/(2 + xi*dt)
fluctuation_size = np.sqrt( grain_mass * kb * temp * xi * half_dt ) # takes dt into account. should it be /grain_mass ?
rescaled_position_step = 2*dt / (2 + xi*dt)

no_fluctuations = False # if True, allows testing for minimum internal energy
fluctuation_factor = 1.0

# boundary conditions
# settings
container_size = 10 # spherical
osmotic_pressure_set = False
soft_vesicle_set = False # if both set False, 'sharp_return' style is used
# boundary force sizes
osmotic_pressure_constant = 1000
soft_vesicle_k = 100
returning_force_mag = 1000


# # # SIMULATION PARMAMETERS # # #
# Run the Monte Carlo algorithm for given number of steps with a progress bar
nsteps = 50000
# Length of Segments, where each segment/grain is 1/5 helical coherence length
coherence_lengths = 20
curved = False
nsegs = 5 * coherence_lengths 
ystart = coherence_lengths/(2*np.pi) if curved else -1*coherence_lengths/2
# Separation, surface to surface (along x axis)
sep = 0.22
#sep += 0.2 # augment for surface to surface
xstartA, xstartB = -sep/2, +sep/2
# Box Limits
xlim, ylim, zlim = 4, 6, 4 # from -lim to +lim, for viewing
boxlims_view = [xlim, ylim, zlim]
# starting shift 
yshift = 0.0


# # # DATA OUTPUT PARAMETERS # # #
# data output directory
mydir = './Data_outputs/test_params/'
if not os.path.exists(mydir):
    os.makedirs(mydir)
# save data
save_data = False
log_update = 100 # how often to publish values to the log file

# terminating settings
recall_steps = 2000
ignore_steps = 2000 + recall_steps
std_tol = 0.01 

# animation
animate = False
frame_hop = 20 # frame dump frequency



# # # Aux functions # # #

# # # ELECTROSTATICS # # #
# next improvement?: interpolation between homologous and non homologous interaction
elstats = Electrostatics()

def interaction_nonhomologous(R_norm, doforce=True):
    '''ASSUMING all non homologous interactions are REPULSIVE ONLY, on the basis that non homologous strands are unlikely to converge'''
    if not doforce: # energy
        #return elstats.calc_a0l0(R_norm*10**-8)
        return 0.0
    elif doforce: # force magnitude
        if R_norm <= 0.2:
            return -1*elstats.calc_da0dR(R_norm*10**-8)
        else:
            return 0.0

def interaction_homologous(R_norm, doforce=True):
    '''For grain with matching dnastr index ONLY (for now)'''
    if not doforce: # energy
        return elstats.find_energy('homol 0', R_norm*10**-8)
    elif doforce: # force magnitude
        return elstats.force('homol 0', R_norm*10**-8)
    
# precomputed interactions for simulation speed
# parameters
precompute_grid_resolution = 2500 # per 0.1 lc
n_Rsteps = int( (R_cut-wall_dist)/0.1 * precompute_grid_resolution )
R_precompute = np.linspace(wall_dist, R_cut, n_Rsteps)

# build lists
force_nonhomologous = []
force_homologous    = []
eng_nonhomologous   = []
eng_homologous      = []
for R_slice in R_precompute:
    force_nonhomologous += [interaction_nonhomologous(R_slice)]
    force_homologous    += [interaction_homologous(R_slice)]
    eng_nonhomologous   += [interaction_nonhomologous(R_slice,doforce=False)]
    eng_homologous      += [interaction_homologous(R_slice,doforce=False)]



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
        
        start_position = []
        for i in range( self.total_points ):
            xi, yi, zi = self.x[i], self.y[i], self.z[i]
            start_position.append( [xi, yi, zi] )
        return np.array(start_position)
    
    def create_strand_straight(self):
        start_position = [ [self.xstart, self.ystart, self.zstart] ]
        for i in range(self.total_points-1):
            start_position.append( [start_position[-1][0], start_position[-1][1]+0.2, start_position[-1][2] ])
        return np.array(start_position)
    
# # # STRAND # # #
# strand and grain attributes
num_segments = nsegs # assume both strands same length
# for StrandA
spA = Start_position(nsegs, xstartA, ystart+yshift, 0)
dnastrA_position = spA.create_strand_curved() if curved else spA.create_strand_straight()
dnastrA_velocity = np.random.choice(np.array([0.0]),size=(num_segments,3))
# for StrandB 
spB = Start_position(nsegs, xstartB, ystart, 0)
dnastrB_position = spB.create_strand_curved() if curved else spB.create_strand_straight()
dnastrB_velocity = np.random.choice(np.array([0.0]),size=(num_segments,3))
        
# data and code efficiency
interactions = [],[] # list[0] is of R_norm, list[1] is of homologous (True) vs nonhomologous (False) 
angle_list_A = []
angle_list_B = []

# Gives list indexes that can break the self interaction lower limit, by interacting ACROSS the strands
cross_list = [] # defined here as to avoid repetitions. 
for i in range(num_segments - self_interaction_limit, num_segments):
    for j in range(i, i+self_interaction_limit+1):
        if j >= num_segments:
            cross_list.append([i,j]) 
    
# bond springs, to hold chain together
@njit
def f_bond(dnastr_position):
    ''''''
    forces = np.random.choice(np.array([0.0]), size=(num_segments,3))
    
    for i in range(num_segments - 1):
        # Vector between consecutive grains
        delta = dnastr_position[i+1] - dnastr_position[i]
        distance = np.linalg.norm(delta)
        force_magnitude = k_spring * (distance - 2*grain_radius)
        force_direction = delta / distance if distance != 0 else np.zeros(3)
        force = force_magnitude * force_direction
        
        forces[i]   += force
        forces[i+1] -= force
    
    return forces
    
@njit     
def f_wlc(dnastr_position):
    '''
    Need to reset angle_list_X before running
    '''
    angle_list_infunc = np.array([0.0],dtype=np.float64)
    
    forces = np.random.choice(np.array([0.0]), size=(num_segments,3))
    
    for i in range(1, num_segments - 1):
    
        # Vectors between adjacent grains
        r1 = dnastr_position[i-1] - dnastr_position[i]
        r2 = dnastr_position[i+1] - dnastr_position[i]
        r1_norm = np.linalg.norm(r1)
        r2_norm = np.linalg.norm(r2)
        # Cosine of the angle between r1 and r2
        cos_theta = np.dot(r1, r2) / (r1_norm * r2_norm)
        theta = np.arccos(cos_theta) #np.arccos(np.clip(cos_theta, -1.0, 1.0)) 
        
        if np.isclose(theta,np.pi):
            angle_list_infunc = np.concatenate( (angle_list_infunc, np.array([0.0]))) 
            forces[i] += np.array([0.0, 0.0, 0.0])
            continue
            
        force_magnitude = -k_bend * (theta - np.pi) 
        force_direction = r1+r2
        force_direction /= np.linalg.norm(force_direction) if np.linalg.norm(force_direction) != 0 else 1
        force = force_magnitude * force_direction
    
        angle_list_infunc = np.concatenate( (angle_list_infunc, np.array([theta-np.pi])))
        forces[i] += force

    return angle_list_infunc[1:], forces
    
# electrostatic interaction  
#@njit      
def f_elstat():
    ''' 
    Reset interactions before
    Takes ALL interactions below the cutoff distance. Only needs calling once per half-step.
    '''
    #interactions_infunc = np.array(([0.0],[0.0]),dtype=np.float64) # return from function to reset & update interactions
    interactions_infunc = [],[]
    forces_A = np.random.choice(np.array([0.0]), size=(num_segments,3))
    forces_B = np.random.choice(np.array([0.0]), size=(num_segments,3)) # return, to add to total force
    
    for i,j in combinations(range(num_segments*2), 2):
        
        if abs(i-j) <= self_interaction_limit and [i,j] not in cross_list:
            continue # skips particular i, j if a close self interaction, avoiding application across strands
        
        # assign grains
        ipos = dnastrA_position[i] if i<num_segments else dnastrB_position[i-num_segments]
        jpos = dnastrA_position[j] if j<num_segments else dnastrB_position[j-num_segments]
        
        # homology of interaction
        ishomol = True if abs(i-j)==num_segments else False
        
        # find intergrain distance
        R = jpos - ipos
        R_norm = np.linalg.norm(jpos - ipos)
        R /= R_norm
        
        # code efficiency 
        if R_norm > R_cut:
            continue # skip this interaction
        
        # save to self.interactions, for efficiency of calculating electrostatic energy
        # done before rescaling R_norm
        #interactions_infunc[0] = np.concatenate( ( interactions_infunc[0], np.array([R_norm]) ) ) 
        #interactions_infunc[1] = np.concatenate( ( interactions_infunc[1], np.array([ishomol]) ) ) 
        interactions_infunc[0].append(R_norm), interactions_infunc[1].append(ishomol)
        
        # code safety, rescale R_norm to avoid dangerous conditions
        #R_norm = wall_dist if R_norm < wall_dist else R_norm # outdated w/ precomputation
        
        # find closest R slice, takes into account wall distance
        close_index = np.argmax(R_norm <= R_precompute)
        
        # linear interaction force
        #f_lin = interaction_homologous(R_norm, doforce=True) if ishomol else interaction_nonhomologous(R_norm, doforce=True)
        f_lin = force_homologous[close_index] if ishomol else force_nonhomologous[close_index]
        
        # apply force
        if i<num_segments:
            forces_A[i] += +1*f_lin*R
        elif i>num_segments:
            forces_B[i-num_segments] += +1*f_lin*R
        if j<num_segments:
            forces_A[j] += -1*f_lin*R
        elif j>num_segments:
            forces_B[j-num_segments] += -1*f_lin*R
            
    return interactions_infunc, forces_A, forces_B
        
# for energies, not including bond springs or translational energy
#@njit
def eng_elstat() -> float:
    '''
    Does electrostatic energy of BOTH strands
    In each step, use after f_elstat() so gen functions do not have to be repeated, homol argument in f_elstat() defines homologicity
    '''
    energy = 0.0
    for p in range(len(interactions[0])):
        # take saved values
        R_norm = interactions[0][p]
        ishomol = interactions[1][p]
        # find closest R slice, note: artifact of wall distance
        close_index = np.argmax(R_norm <= R_precompute)
        # add to energy
        #energy += interaction_homologous(R_norm, doforce=False) if ishomol else interaction_nonhomologous(R_norm, doforce=False) # outdated method
        energy += eng_homologous[close_index] if ishomol else eng_nonhomologous[close_index]
    return energy / (kb*300) # give energy in kbT units
   
@njit     
def eng_elastic(angle_list) -> float:
    '''
    MUST be run in loop, for i in (1, num_segments-1)
    Energy term for bending of DNA strand from straight
    Uses small angle approx so finer coarse graining more accurate
    '''
    energy = 0.0
    for angle in angle_list:
        energy += 1/2 * k_bend *  angle**2
    return energy / (kb*300) # give energy in kbT units
    
    
# # # SIMULATION # # #
# data
trajectoryA = []
trajectoryB = []
energy_traj = []
endtoend_traj = []
mean_curvature_traj = []
std_curvature_traj = []
total_pairs_traj = []
homol_pairs_traj = []
homol_pair_dist_traj = []
terminal_dist_traj = []
n_islands_traj = []
        
        
def langevin_fluctuations(fluctuation_factor):
    '''
    Build list of fluctuations for each grain. Correlated within each 'correlation length'.
    Correlation length usually kuhn length (25) or, less often, helical coherence length (5)
    
    Applies UNcorrellated brownian force to each 'correlation length'
    Fully correllated within fluctuation 'correlation length'
    '''
    # if no fluctuations specified (for bug testing), return array of zeroes
    if no_fluctuations:
        return np.zeros(num_segments*2)
    
    # reset lists
    fluctuationA , fluctuationB = [] , []
    
    # build lists, may have different lengths
    adjusted_fluctuation_size = fluctuation_size * fluctuation_factor
    for i in range(int(np.ceil(num_segments/correlation_length))):
        fluctuationA += [(np.random.normal(0, adjusted_fluctuation_size, size=3) )] * correlation_length
    for i in range(int(np.ceil(num_segments/correlation_length))):
        fluctuationB += [(np.random.normal(0, adjusted_fluctuation_size, size=3) )] * correlation_length
    
    # correct length
    fluctuationA, fluctuationB = fluctuationA[:num_segments] , fluctuationB[:num_segments] 
    
    # put together, for form taken by functions
    fluctuation_list = fluctuationA + fluctuationB
    
    return fluctuation_list
        

def apply_box(dnastr_position):
    '''
    Constant force be applied to entire strand when one part strays beyond box limits
    
    Update required: different boundary conditions
        - soft walls (weak spring)
        - osmotic pressure (constant central force)
    '''
    if osmotic_pressure_set: # independent of container size
        # calculate vector from centre of mass towards centre of box (0, 0, 0)
        centre_mass_vec = np.mean(dnastr_position,axis=0)
        centre_mass_vec /= np.linalg.norm(centre_mass_vec)
        # apply osmotic pressure to all grains
        return - osmotic_pressure_constant * centre_mass_vec
    
    if soft_vesicle_set: # spherical, returning force to centre, only largest boxlim matters
        dnastr_position_norms = np.linalg.norm(dnastr_position,axis=1)
        return (dnastr_position_norms > container_size) * -soft_vesicle_k * (dnastr_position / dnastr_position_norms)
    
    if not osmotic_pressure_set and not soft_vesicle_set: # sharp return style, spherical
        centre_mass_vec = np.mean(dnastr_position,axis=0)
        centre_mass_vec /= np.linalg.norm(centre_mass_vec)
        return returning_force_mag * centre_mass_vec * (np.linalg.norm(dnastr_position,axis=1).any() > container_size)


# for data analysis
def record():
    
    trajectoryA.append(dnastrA_position)
    trajectoryB.append(dnastrB_position)
    
    energy_traj.append( eng_elastic(angle_list_A) + eng_elastic(angle_list_B) + eng_elstat() )
    
    endtoend_traj.append(find_endtoend(-1))
    
    mean_curvature_traj.append( np.mean( angle_list_A + angle_list_B ) )
    std_curvature_traj.append( np.std( angle_list_A + angle_list_B ) )
    
    total_pairs, homol_pairs, homol_pair_dist, terminal_dist, n_islands = find_pair_data()
    total_pairs_traj.append(total_pairs)
    homol_pairs_traj.append(homol_pairs)
    homol_pair_dist_traj.append(homol_pair_dist)
    terminal_dist_traj.append(terminal_dist)
    n_islands_traj.append(n_islands)
    
    
def find_pair_data():
    # pair numbers
    total_pairs = len(interactions[0])
    homol_pairs_tot = np.sum(interactions[1])
    
    # find homologous pair distances
    homol_R_norm_list = np.delete(interactions[0],np.array(interactions[1])==False)
    homol_pair_dist = np.mean(homol_R_norm_list) if len(homol_R_norm_list)>0 else None
    homol_pairs = np.sum( homol_R_norm_list < np.array([0.25]) )
    
    # find number of 'islands'
    homol_R_norm_list = np.linalg.norm(np.array(dnastrA_position) - np.array(dnastrB_position), axis=1)
    n_islands = len( signal.find_peaks( np.concatenate( ( np.array([False]), (homol_R_norm_list < R_cut) , np.array([False]) ) ) )[0] )
    
    # separation of end pairs
    terminal_dist = np.mean( [abs(dnastrA_position[0] - dnastrB_position[0]), abs(dnastrA_position[-1] - dnastrB_position[-1])] )
    
    return total_pairs, homol_pairs, homol_pair_dist, terminal_dist, n_islands


def find_endtoend(tindex):
    endtoendA = np.linalg.norm(trajectoryA[tindex][0] - trajectoryA[tindex][-1]) + 0.2
    endtoendB = np.linalg.norm(trajectoryB[tindex][0] - trajectoryB[tindex][-1]) + 0.2 # account for size of particle
    return endtoendA, endtoendB
     

# # # RUN SIMULATION # # # 

# logging
for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
log_filename = datetime.now().strftime(mydir+'LOG_%Y%m%d_%H%M%S.log')
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')
logging.info('Simulation started')
logging.info(f'''Simulation parameters:
    gamma    : {gamma}
    xi       : {xi}
    dt       : {dt}
    nsteps   : {nsteps}
    num parts: {nsegs}
    num l_c  : {coherence_lengths}
    container: {container_size}
Starting conditions:
    separation: {sep} (central)
    curvature : {curved}
             ''')
             
for i in range(nsteps):
    
    # for time integration / evolution
    # RUN STEP CODE - PUT IN LANGEVIN RUN
    # reset and calculate external forces
    # reset
    dnastrA_extforce = np.random.choice(np.array([0.0]),size=(num_segments,3))
    dnastrB_extforce = np.random.choice(np.array([0.0]),size=(num_segments,3))

    # calculate
    # bonds
    dnastrA_extforce += f_bond(dnastrA_position)
    dnastrB_extforce += f_bond(dnastrB_position)

    # WLC
    angle_list_A, add_forceA = f_wlc(dnastrA_position)
    angle_list_B, add_forceB = f_wlc(dnastrB_position)
    dnastrA_extforce += add_forceA
    dnastrB_extforce += add_forceB

    # Electrostatics
    interactions, add_forceA, add_forceB = f_elstat()
    dnastrA_extforce += add_forceA
    dnastrB_extforce += add_forceB

    # Boundaries
    add_forceA = apply_box(dnastrA_position)
    add_forceB = apply_box(dnastrB_position)
    dnastrA_extforce += add_forceA
    dnastrB_extforce += add_forceB

    # take random fluctuation length for each 'correlation length'
    fluctuation_list = langevin_fluctuations(fluctuation_factor)

    # update velocities for first half step
    dnastrA_velocity += half_dt * dnastrA_extforce / grain_mass
    dnastrA_velocity += fluctuation_list[:num_segments]
    dnastrB_velocity += half_dt * dnastrB_extforce / grain_mass
    dnastrB_velocity += fluctuation_list[num_segments:]

    # update positions
    dnastrA_position += dnastrA_velocity * rescaled_position_step
    dnastrB_position += dnastrB_velocity * rescaled_position_step

    # update velocities for second half step
    dnastrA_velocity *= applied_friction_coeff
    dnastrA_velocity += half_dt * dnastrA_extforce / grain_mass
    dnastrA_velocity += fluctuation_list[:num_segments]
    dnastrB_velocity *= applied_friction_coeff
    dnastrB_velocity += half_dt * dnastrB_extforce / grain_mass
    dnastrB_velocity += fluctuation_list[num_segments:]

    # save data
    record()
    
    # progress bar
    length = 20
    progress = (i + 1) / nsteps
    bar_length = int(length * progress)
    bar = f"[{'=' * bar_length:{length}}] {progress * 100:.1f}%"
    sys.stdout.write(f"\r{bar}")
    sys.stdout.flush()
    
    # log file
    if i % log_update == 0:
        endtoendA, endtoendB = endtoend_traj[-1][0], endtoend_traj[-1][1]
        logging.info(f'''Step {i} : DATA:
Simulation Internal Energy = {energy_traj[-1]}

Strand A end to end = {endtoendA} lc
Strand B end to end = {endtoendB} lc
Mean Curvature      = {mean_curvature_traj[-1]*180/np.pi} degrees
STD  Curvature      = {std_curvature_traj[-1]*180/np.pi} degrees

Total Pairs              = {total_pairs_traj[-1]}
Homologous Pairs         = {homol_pairs_traj[-1]}
Homologous Pair Distance = {homol_pair_dist_traj[-1]} lc
Number Islands           = {n_islands_traj[-1]}

...''')
        print(f'''\rSimulation Internal Energy = {energy_traj[-1]}
Strand A end to end      = {endtoendA} lc
Strand B end to end      = {endtoendB} lc
Mean Curvature           = {mean_curvature_traj[-1]*180/np.pi} degrees
STD  Curvature           = {std_curvature_traj[-1]*180/np.pi} degrees
Total Pairs              = {total_pairs_traj[-1]}
Homologous Pairs         = {homol_pairs_traj[-1]}
Homologous Pair Distance = {homol_pair_dist_traj[-1]} lc
Number Islands           = {n_islands_traj[-1]}
...''')
        
        # Stopping conditions
        if not endtoendA < 50*coherence_lengths and not endtoendA > 50*coherence_lengths or not endtoendB < 50*coherence_lengths and not endtoendB > 50*coherence_lengths: #always True if 'nan'
            error_msg = f'STEP {i}: Simulation terminating - lost grains'
            print(error_msg)
            logging.info(error_msg)
            break # end simulation
            
        if i>ignore_steps and np.std( homol_pairs_traj[-recall_steps:] )/num_segments < std_tol:
            finish_msg = f'STEP {i}: Simulation terminating - pairs converged'
            print(finish_msg)
            logging.info(finish_msg)
            break # end simulation
    
# save trajectories
if save_data:
    with open(mydir+'test_simulation.dat','wb') as data_f:
        pickle.dump([trajectoryA, trajectoryB], data_f)
        

# extracting data from trajectories
xsteps = np.linspace(0,len(trajectoryA),len(trajectoryA))
endtoendA, endtoendB = np.array(endtoend_traj)[:,0], np.array(endtoend_traj)[:,1]
endtoendA, endtoendB = list(endtoendA), list(endtoendB)

# plotting end to end distances
endtoendendA, endtoendendB = find_endtoend(-1)
print()
print(f'End to end distance Strand A = {endtoendendA}')
print(f'End to end distance Strand B = {endtoendendB}')

plt.figure()
plt.title('Coarse Grain DNA End to End Distance')
plt.xlabel(f'Timestep, {dt}')
plt.ylabel('End to End distance, $l_c$')
plt.plot(xsteps, endtoendA, label = 'Strand A')
plt.plot(xsteps, endtoendB, label = 'Strand B')
plt.grid(linestyle=':')
plt.legend(loc='best')
plt.savefig(mydir+'endtoend.png')
plt.show()

# plotting total internal energy
print()
print(f'Internal Energy = {energy_traj[-1]} kbT')

plt.figure()
plt.title('Coarse Grain DNA Internal Energy')
plt.xlabel(f'Timestep, {dt}')
plt.ylabel('Energy, $k_bT$')
plt.plot(xsteps, energy_traj)
plt.grid(linestyle=':')
plt.savefig(mydir+'energy.png')
plt.show()

# plotting curvature
print()
print(f'Mean Curvature = {mean_curvature_traj[-1]*180/np.pi} degrees')
print(f'STD  Curvature = {std_curvature_traj[-1]*180/np.pi} degrees')

plt.figure()
plt.title('Mean Curvature')
plt.xlabel(f'Timestep, {dt}')
plt.ylabel('Curvature, degrees')
plt.plot(xsteps, abs(np.array(mean_curvature_traj)*180/np.pi) + np.array(std_curvature_traj)*180/np.pi, label='+std', color='orange')
plt.plot(xsteps, abs(np.array(mean_curvature_traj)*180/np.pi), label='mean')
plt.plot(xsteps, abs(np.array(mean_curvature_traj)*180/np.pi) - np.array(std_curvature_traj)*180/np.pi, label='-std', color='orange')
plt.grid(linestyle=':')
plt.legend(loc='best')
plt.savefig(mydir+'curvature.png')
plt.show()

# plotting pair data
print()
print(f'Total Pairs = {total_pairs_traj[-1]}') # may want to remove
print(f'Homologous Pairs = {homol_pairs_traj[-1]}')
print(f'Homologous Pair Distance = {homol_pair_dist_traj[-1]} lc')
print(f'Number islands = {n_islands_traj[-1]}')

plt.figure()
plt.title('Pair Number')
plt.xlabel(f'Timestep, {dt}')
plt.ylabel('Number of Pairs')
plt.plot(xsteps, np.array(total_pairs_traj)-np.array(homol_pairs_traj), label='Non Homol')
plt.plot(xsteps, homol_pairs_traj, label='Homologous')
plt.grid(linestyle=':')
plt.legend(loc='best')
plt.savefig(mydir+'pair_counts.png')
plt.show()

plt.figure(figsize=[16,5])

plt.subplot(1, 2, 1)
plt.title('Pair Number')
plt.xlabel(f'Timestep, {dt}')
plt.ylabel('Number of Pairs')
plt.plot(xsteps, n_islands_traj, label='Islands')
plt.plot(xsteps, homol_pairs_traj, label='Homologous Pairs')
plt.grid(linestyle=':')
plt.legend(loc='best')

plt.subplot(1, 2, 2)
plt.title('Homologous Pair Separation')
plt.xlabel(f'Timestep, {dt}')
plt.ylabel('Distance, $l_c$')
plt.plot(xsteps, homol_pair_dist_traj, label='All homologous pairs')
plt.plot(xsteps, terminal_dist_traj, label='End homologous pairs')
plt.grid(linestyle=':')
plt.legend(loc='best')

plt.savefig(mydir+'islands.png')
plt.show()
    
# # # ANIMATION # # #
if animate:
    # data
    tA = np.array(trajectoryA)
    tB = np.array(trajectoryB)
    
    # Number of frames for the animation
    selected_frames = range(0,len(tA),frame_hop)
    num_frames = len(selected_frames)
    
    # Create a figure and a 3D axis
    fig = plt.figure(figsize=[16,10])
    ax = fig.add_subplot(111, projection='3d')
    
    # Precomputed data (replace these lists with your actual data)
    # Each element in the list should be a tuple (x, y, z)
    data1 = [(tA[i][:, 0], tA[i][:, 1], tA[i][:, 2]) for i in selected_frames]
    data2 = [(tB[i][:, 0], tB[i][:, 1], tB[i][:, 2]) for i in selected_frames]
    
    # Initial data
    x1, y1, z1 = data1[0]
    x2, y2, z2 = data2[0]
    
    # Initial line plots
    line1, = ax.plot(x1, y1, z1, color='b', marker='.', markersize=1, label='DNA strand A')
    line2, = ax.plot(x2, y2, z2, color='r', marker='.', markersize=1, label='DNA strand B')
    
    # Create a text label for the frame number, initially set to the first frame
    frame_text = ax.text2D(0.05, 0.95, f"Frame: {selected_frames[0]}", transform=ax.transAxes)
    
    # Axis limits
    ax.set_xlim(-boxlims_view[0], boxlims_view[0])
    ax.set_ylim(-boxlims_view[1], boxlims_view[1])
    ax.set_zlim(-boxlims_view[2], boxlims_view[2])
    ax.legend()
    
    # Update function for the animation
    def update(frame):
        # Fetch data for the current frame
        x1, y1, z1 = data1[frame]
        x2, y2, z2 = data2[frame]
        
        # Update line plots with new data
        line1.set_data_3d(x1, y1, z1)
        line2.set_data_3d(x2, y2, z2)
        
        # Update the frame number text
        frame_text.set_text(f"Frame: {selected_frames[frame]}")

        return line1, line2, frame_text
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)
    
    # Save the animation as an MP4 file (uncomment to save)
    ani.save(mydir+'3d_line_animation.gif', writer=PillowWriter(fps=20))
    
    logging.info('Animation saved as gif')
    
    # Show the plot
    plt.show()
    
logging.info('Simulation completed')