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

# Truncated LJ excluded volume interaction
sigma = 0.10 # distance scale for excluded volume interactions, where Grain diameter is 0.2
epsilon = kb * 300 # energy scale for excluded volume interactions

# NON-homologous pairs interaction energy, UNITS TO COMPARE??
x = np.linspace(0,1000,int(1000/0.2)+1)
nonhomolfunc = np.concatenate((-x[x <= 1.0],(x-2)[x > 1])) #zero@start, -1 kbT @ 1 lc, 0 kbT @ 2lc, & so on

# Homologous pairs interaction energy, for grains 1/5 the corellation length
homologous_pair_interaction = -4/5 * kb * 300

# Langevin 
lamb = 1 # damping coefficient
dt = 0.002 # timestep

# # # Aux functions # # #
def gen_grains(coherence_lengths, start_position):
    strand = [Grain(start_position, np.zeros(3) )]
    for i in range(5*coherence_lengths):
        strand.append( Grain( strand[-1].position + np.array([0, 0.2, 0]), np.zeros(3) ) )
    return strand

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
    
    def overlap(self, other) -> tuple([bool, float]):
        inter_vector = self.position - other.position
        dist = np.linalg.norm(inter_vector)
        equilib = self.radius + other.radius + 0.01
        return dist <= equilib, dist, inter_vector


class Strand:
    def __init__(self, grains):
        self.dnastr = grains
        self.num_segments = len(grains)
        self.interactivity = []
        self.identities = []
            
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
            #print(' bond force:  ',force)
    
    # for excluded volume interactions
    def f_excvol(self, other):
        for i, j in combinations(range(len(self.dnastr+other.dnastr)),2):
            if abs(i-j) == 1 and i in [self.num_segments-1,self.num_segments] and j in [self.num_segments-1,self.num_segments]: # ignoring nearest neighbours in excluded volume interactions
                continue # skips particular i, j
            igrain = self.dnastr[i] if i<self.num_segments else other.dnastr[i-self.num_segments] 
            jgrain = self.dnastr[j] if j<self.num_segments else other.dnastr[j-self.num_segments] # use correct strand
            r = jgrain.position - igrain.position
            r_norm = np.linalg.norm(r)
            if r_norm < sigma and r_norm != 0:
                # Lennard-Jones potential (repulsive part)
                force_magnitude = 24 * epsilon * (sigma / (r_norm + sigma))**6 / (r_norm + sigma) # Soft core parameter to avoid divergence, 1 kbT for cores overlap
                force_direction = r / r_norm
                force = force_magnitude * force_direction
                #self.dnastr[i].update_velocity(-force, 1) if i<self.num_segments else self.dnastr[i-self.num_segments].update_velocity(-force, 1)
                #self.dnastr[j].update_velocity( force, 1) if j<self.num_segments else self.dnastr[j-self.num_segments].update_velocity( force, 1)
                igrain.update_velocity(-force, dt)
                jgrain.update_velocity( force, dt)
                #print(' excvol force:',force)
                
    # WLC bending forces
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
            #print(' wlc torque:  ',torque)
        
    def find_angle(self, seg_index):
        p1 = self.dnastr[seg_index-1].position
        p2 = self.dnastr[seg_index].position
        p3 = self.dnastr[seg_index+1].position
        
        # colinear vectors
        if np.isclose(p3-p2,p2-p1).all():
            return 0
        # return 180 - angle
        return np.pi - np.arccos(np.dot(p3-p2,p1-p2) / (np.linalg.norm(p3-p2)*np.linalg.norm(p1-p2) ) )
    
    # for conditional electrostatic interaction
    def count_adj_same(self, index: int, identify=False) -> int:
        '''
        For an index, counts number of paired segments within the same DNA strand.
        Considering ALL adjacent sites, otherwise count does not work.
        '''
        count = 0 # will NOT count self and neighbours
        identity = []
        for bi in range(self.num_segments):
            if abs(bi-index) > 1:
                count += 1 if self.dnastr[index].overlap(self.dnastr[bi])[0] else 0
                if identify:
                    identity += ['s'+str(bi)]
        return count, identity
    
    def count_adj_other(self, selfindex, other, identify=False) -> int:
        '''
        For an index, counts number of paired segments with the other DNA strand.
        For single specified segment only.
        '''
        count = 0 # WILL not count self and neighbours
        identity = []
        for bi in range(self.num_segments):
            if abs(bi-selfindex) > 1:
                count += 1 if self.dnastr[selfindex].overlap(other.dnastr[bi])[0] else 0
                if identify:
                    identity += ['o'+str(bi)]
        return count, identity
    
    def count_all(self, strand1, strand2) -> tuple([int, int, int]):
        '''Counts ALL pairings across strands, with any input strand - allowing use for provisional'''
        count_other = 0
        for b1 in strand1.dnastr:
            for b2 in strand2.dnastr:
                count_other += 1 if b1.overlap(b2)[0] else 0
        count_same = 0
        for i, j in combinations(range(self.num_segments),2):
            if abs(i-j) == 1:
                continue
            count_same += 1 if strand1.dnastr[i].overlap(strand1.dnastr[j])[0] else 0
            count_same += 1 if strand2.dnastr[i].overlap(strand2.dnastr[i])[0] else 0
        return count_other + count_same, count_other, count_same
    
    def condition_interactivity(self, other, fororbac: int, first: bool, seg_index: int, num = 0) -> Tuple[int]:
        '''Defines an interaction as attractive (-1) if it is 'standalone', otherwise repulsive (1) or no interaction (0)
        '''
        count_s, identity_s = self.count_adj_same(seg_index)
        count_o, identity_o = self.count_adj_other(seg_index, other)
        count = count_s + count_o
        identities = identity_s + identity_o
        if first:
            if count > num:
                return [1],identities
            else:
                return [0],identities
    
        if count > num and abs(self.interactivity[fororbac]) == 0:
            return [1],identities
        elif count > num and abs(self.interactivity[fororbac]) != 0: 
            return [abs(self.interactivity[fororbac])+1], identities
        else:
            return [0], identities
        
    def gen_interactivity(self, other) -> list:
        '''
        Generates from 0th Bead, electrostatic interaction counted for whole Strand
        Does for BOTH strands
        '''
        self.interactivity,  self.identities  = self.condition_interactivity(other, 0, True, 0, 0)
        other.interactivity, other.identities = other.condition_interactivity(self, 0, True, 0, 0)
        for seg_index in range(1,self.num_segments): # from index+1 to penultimate
            inter, iden = self.condition_interactivity(other, -1, False, seg_index, 0) # forward
            self.interactivity += inter
            self.identities += [iden]
            inter, iden = other.condition_interactivity(self, -1, False, seg_index, 0)
            other.interactivity += inter
            other.identities += [iden]
        
    def gen_interactivity_homol(self, other):
        '''For homologous strands, must be used after "gen_interactivity" to overwrite non-homol interactions '''
        for i, j in combinations(range(len(self.dnastr+other.dnastr)),2):
            # rescale i, j
            inew = i if i>self.num_segments else i-self.num_segments
            jnew = j if j>self.num_segments else j-self.num_segments
            # use correct strand
            igrain = self.dnastr[i] if i<self.num_segments else other.dnastr[inew] 
            jgrain = self.dnastr[j] if j<self.num_segments else other.dnastr[jnew]
            if abs(inew-jnew) <= 15 and igrain.overlap(jgrain)[0]: # homologous recognition funnel
                self.interactivity[inew] = homologous_pair_interaction/abs(inew-jnew) if i < self.num_segments else self.interactivity[inew]
                other.interactivity[inew] = homologous_pair_interaction/abs(inew-jnew) if i > self.num_segments else other.interactivity[inew]
                self.interactivity[jnew] = homologous_pair_interaction/abs(inew-jnew) if j < self.num_segments else self.interactivity[jnew]
                other.interactivity[jnew] = homologous_pair_interaction/abs(inew-jnew) if j > self.num_segments else other.interactivity[jnew]
        
    def apply_interactivity(self, other, pindex):
        '''
        Auxillary function for f_elstat(), applies the veloctities for grains backwards from peak
        NOTE: may be faster just to use a loop of all combinations
        '''
        for i in range(pindex, pindex-self.interactivity[pindex], -1): # looking backwards
                for n in self.identities[i]:
                    felstat = -1 if self.interactivity[i] <= 5 else +1 # attractive if within helical coherence length
                    # identities provides the information of each strand and index for the vector
                    fvec = self.dnastr[i].overlap(self.dnastr[int(self.identities[i][n][1])])[3] if self.identities[i][n][0] == 's' else self.dnastr[i].overlap(other.dnastr[int(self.identities[i][n][1])])[3]
                    self.dnastr[i].update_velocity(felstat*fvec,dt)
        if self.interactivity[-1] != [0]: # final list element, will be missed by scipy.signal.find_peaks()
            for i in range(pindex, pindex-self.interactivity[-1], -1):
                for n in self.identities[i]:
                    felstat = -kb*300 if self.interactivity[i] <= 5 else +kb*300 # attractive if within helical coherence length
                    # identities provides the information of each strand and index for the vector
                    fvec = self.dnastr[i].overlap(self.dnastr[int(self.identities[i][n][1])])[3] if self.identities[i][n][0] == 's' else self.dnastr[i].overlap(other.dnastr[int(self.identities[i][n][1])])[3]
                    force = felstat * fvec * 300 # put in kb (not kbT) for comparison with f_wlc
                    self.dnastr[i].update_velocity(force,dt)
                    #print(' elstat force:',felstat*fvec)
                    
    
    def f_elstat(self, other, homol=False):
        # find each interacting segment from start to end
        # apply attraction or repulsion to all (based off gradient of non homologous energy) ?
        # OR choose attractive grain, repel all others by increasing amount ?
        # INCLUDE variation by distance - like trunctated LJ force - cutoff VS particle size??
        # NOTE: if cutoff is significantly larger, increase num for counting pairs
        # -delE of function from recent paper
        ''' '''
        self.gen_interactivity(other)
        if homol:
            self.gen_interactivity_homol(other)
        # for self strand
        for p in signal.find_peaks(self.interactivity)[0]: # much shorter loop
            self.apply_interactivity(other, p)
        # for other strand
        for p in signal.find_peaks(other.interactivity)[0]:
            other.apply_interactivity(self, p)
        
    # for energies, not including bond springs or translational energy
    def eng_elstat(self, other, homol=False):
        '''Does electrostatic energy of BOTH strands, avoids lengthy loops of eng_elec_old'''
        self.gen_interactivity(other)
        if homol:
            self.gen_interactivity_homol(other)
        energy = 0
        for i in signal.find_peaks(self.interactivity)[0]: # much shorter loop
            energy += nonhomolfunc[self.interactivity[i]]
        energy += nonhomolfunc[self.interactivity[-1]] # SciPy will miss any final
        for i in signal.find_peaks(other.interactivity)[0]: # much shorter loop
            energy += nonhomolfunc[other.interactivity[i]]
        energy += nonhomolfunc[other.interactivity[-1]] # SciPy will miss any final
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
        self.StrandA.f_excvol(self.StrandB)
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
            random_force = np.random.normal(0, np.sqrt(2 * lamb * kb * temp), size=3) #/dt
            # Damping force
            grain.update_velocity(random_force,dt)
            #print(' drag force:   ',damping_force)
            #print(' rand force:   ',random_force)
            
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
        comparearray = np.zeros(len(self.StrandA.interactivity))
        pairsA = np.sum(np.array(self.StrandA.interactivity)!=comparearray)
        pairsB = np.sum(np.array(self.StrandB.interactivity)!=comparearray)
        totpairs = int((pairsA+pairsB)/2)
        return totpairs, abs(pairsA - pairsB)  
          
    def find_energy(self):
        '''Includes electrostatic and WLC bending energies ONLY'''
        return self.StrandA.eng_elastic() + self.StrandB.eng_elastic() + self.StrandA.eng_elstat(self.StrandB)
    
