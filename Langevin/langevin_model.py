# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 18:11:06 2024

@author: Ralph Holden
"""
# imports
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Tuple

# # # UNITS # # #
kb = 1.38e-23
temp = 310.15

# # # PARAMETERS # # #
# Worm Like Chain Bending
lp = 4.5 # persistence length, in coherence length diameter grains of 100 Angstroms
kappab = lp * kb * temp # bending stiffness
s = 0.4 # standard distance through chain separated by one Grain
k_bend = -kappab/s  # Bending stiffness constant

# Truncated LJ excluded volume interaction
sigma = 0.15 # distance scale for excluded volume interactions, where Grain diameter is 0.2
epsilon = 5*kb # energy scale for excluded volume interactions

# NON-homologous pairs interaction energy, UNITS TO COMPARE??
x = np.linspace(0,1000,int(1000/0.2)+1)
nonhomolfunc = np.concatenate((-x[x <= 1.0],(x-2)[x > 1])) #zero@start, -1 kbT @ 1 lc, +1 kbT @ 2lc, & so on

# Homologous pairs interaction energy
homolfunc = -x 

# Langevin 
lamb = 1 # damping coefficient
dt = 0.001 # timestep

# # # Aux functions # # #
def rand_v():
    return [np.random.random()*[-0.1,0.1][np.random.randint(2)],np.random.random()*[-0.1,0.1][np.random.randint(2)],np.random.random()*[-0.1,0.1][np.random.randint(2)]]

def gen_grains(coherence_lengths, start_position):
    strand = [Grain(start_position, rand_v())]
    for i in range(5*coherence_lengths -1):
        strand.append( Grain( strand[-1].position + np.array([0,0.2,0]), rand_v() ) )
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
    
    def overlap(self, other,) -> tuple([bool, float]):
        inter_vector = self.position-other.position
        dist = np.linalg.norm(inter_vector)
        equilib = self.radius + other.radius
        return dist <= equilib, dist


class Strand:
    def __init__(self, grains):
        self.dnastr = grains
        self.num_segments = len(grains)
            
    def copy(self):
        return (Strand(grains = self.dnastr))
    
    # bond springs, to hold chain together
    def f_bond(self):
        k_spring = 100.0  # Spring constant for bonds
        for i in range(self.num_segments - 1):
            # Vector between consecutive grains
            delta = self.dnastr[i+1].position - self.dnastr[i].position
            distance = np.linalg.norm(delta)
            force_magnitude = k_spring * (distance - 2*self.dnastr[0].radius)
            force_direction = delta / distance if distance != 0 else np.zeros(3)
            force = force_magnitude * force_direction
            self.dnastr[i].update_velocity(force, 1)
            self.dnastr[i+1].update_velocity(-force, 1)
    
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
                force_magnitude = 24 * epsilon * (2 * (sigma / r_norm)**13 - (sigma / r_norm)**7) / r_norm
                force_direction = r / r_norm
                force = force_magnitude * force_direction
                #self.dnastr[i].update_velocity(-force, 1) if i<self.num_segments else self.dnastr[i-self.num_segments].update_velocity(-force, 1)
                #self.dnastr[j].update_velocity( force, 1) if j<self.num_segments else self.dnastr[j-self.num_segments].update_velocity( force, 1)
                igrain.update_velocity(-force, dt)
                jgrain.update_velocity( force, dt)
                
    # WLC bending energies
    def f_wlc(self):
        for i in range(1, self.num_segments - 1):
            # Vectors between adjacent grains
            r1 = self.dnastr[i].position - self.dnastr[i-1].position
            r2 = self.dnastr[i+1].position - self.dnastr[i].position
            r1_norm = np.linalg.norm(r1)
            r2_norm = np.linalg.norm(r2)
            if r1_norm == 0 or r2_norm == 0:
                continue
            # Cosine of the angle between r1 and r2
            cos_theta = np.dot(r1, r2) / (r1_norm * r2_norm)
            theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            torque_magnitude = k_bend * (theta - np.pi) / (r1_norm * r2_norm)
            torque_direction = np.cross(r1, r2)
            torque_direction /= np.linalg.norm(torque_direction) if np.linalg.norm(torque_direction) != 0 else 1.0
            torque = torque_magnitude * torque_direction
            self.dnastr[i].update_velocity(torque, dt)
            self.dnastr[i+1].update_velocity(-torque, dt)
        
    def find_angle(self, seg_index):
        p1 = self.dnastr[seg_index-1].position
        p2 = self.dnastr[seg_index].position
        p3 = self.dnastr[seg_index+1].position
        
        # Check if the points are collinear by calculating the area of the triangle formed by the points
        # If the area is zero, the points are collinear
        if np.isclose(np.linalg.det(np.array([p1 - p3, p2 - p3, p3 - p3])),0):
            return 0, 1, 1 # will give 0 angular energy
        
        mid1 = (p1 + p2) / 2
        mid2 = (p2 + p3) / 2
        
        # Vectors
        vec1 = p2 - p1
        vec2 = p3 - p2
    
        # Perpendicular vectors in the plane formed by the points
        perp1 = np.cross(vec1, np.array([1, 0, 0]) if np.abs(vec1[0]) < np.abs(vec1[1]) else np.array([0, 1, 0]))
        perp1 /= np.linalg.norm(perp1)
    
        perp2 = np.cross(vec2, np.array([1, 0, 0]) if np.abs(vec2[0]) < np.abs(vec2[1]) else np.array([0, 1, 0]))
        perp2 /= np.linalg.norm(perp2)
    
        # Solve the plane equations for the intersection
        A = np.array([perp1, -perp2]).T
        B = np.array([np.dot(perp1, mid1), np.dot(perp2, mid2)])
        
        centre = np.linalg.solve(A, B)
        
        s = np.linalg.norm(p1 - p3)
        r = np.linalg.norm(p1 - centre) # radius of circle
        
        angle = 2*np.pi*r / s # fraction of circle circumference is angle in rad
        
        return angle, s, r
    
    # for conditional electrostatic interaction
    def count_adj_same(self, index: int) -> int:
        '''
        For an index, counts number of paired segments within the same DNA strand.
        Considering ALL adjacent sites, otherwise count does not work.
        '''
        count = 0 # WILL count neighbours and itself, expect 3 per mid-chain grain
        for a in self.dnastr:
            if self.dnastr[index].overlap(a)[0]:
                count += 1
        return count
    
    def count_adj_other(self, selfindex, other) -> int:
        '''
        For an index, counts number of paired segments with the other DNA strand.
        For single specified segment only.
        '''
        count = 0 
        for b in other.dnastr:
            if self.dnastr[selfindex].overlap(b)[0]:
                count += 1 
        return count
    
    def condition_interactivity(self, other, fororbac: int, first: bool, seg_index: int, num = 3) -> Tuple[int]:
        '''Defines an interaction as attractive (-1) if it is 'standalone', otherwise repulsive (1) or no interaction (0)
        '''
        if first:
            return [1] if self.count_adj_same(seg_index) + self.count_adj_other(seg_index, other) > num else [0]

        if self.count_adj_same(seg_index) + self.count_adj_other(seg_index, other) > num and abs(self.interactivity[fororbac]) == 0:
            return [1]
        elif self.count_adj_same(seg_index) + self.count_adj_other(seg_index, other) > num and abs(self.interactivity[fororbac]) != 0: 
            return [abs(self.interactivity[fororbac])+1] # more repulsive than previous by kbT
        else:
            return [0]
        
    def gen_interactivity(self, other) -> list:
        '''Prioritises sticking to middle (first assigned hence without dependance on +- 1)
        Generates from 0th Bead, electrostatic interaction counted for whole Strand
        Does for BOTH strands
        '''
        self.interactivity  = self.condition_interactivity(other, 0, True, 0, 2) # starter
        other.interactivity = other.condition_interactivity(self, 0, True, 0, 2)
        for seg_index in range(1,self.num_segments-1): # from index+1 to penultimate
            self.interactivity  += self.condition_interactivity(other, -1, False, seg_index, 3) # forward
            other.interactivity += other.condition_interactivity(self, -1, False, seg_index, 3)
        self.interactivity  += self.condition_interactivity(other, -1, False, -1, 2) # end
        other.interactivity += other.condition_interactivity(self, -1, False, -1, 3)
        
    def f_elstat(self):
        pass
        # find each interacting segment from start to end
        # apply attraction or repulsion to all (based off gradient of non homologous energy) ?
        # OR choose attractive grain, repel all others by increasing amount ?
        # INCLUDE variation by distance - like trunctated LJ force
        
    # for energies, not including bond springs or translational energy
    def eng_elec(self, other):
        '''Does electrostatic energy of BOTH strands, avoids lengthy loops of eng_elec_old'''
        self.gen_interactivity(other)
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
            angle, s, r = self.find_angle(seg_index)
            energy += kappab / (2*s) *  angle**2
        return energy
    
class Simulation:
    def __init__(self, StrandA: Strand, StrandB: Strand, boxlims: np.array):
        self.StrandA = StrandA
        self.StrandB = StrandB
        
        self.boxlims = boxlims # boxlims a Vector
    
        # data
        self.trajectoryA = []
        self.trajectoryB = []
        self.pair_counts = []
        self.energies = []
        self.endtoends = []
        #self.record()

    def run_step(self):
        self.apply_langevin_dynamics()
        self.StrandA.f_bond(), self.StrandB.f_bond()
        self.StrandA.f_wlc(), self.StrandB.f_wlc() 
        self.StrandA.f_excvol(self.StrandB)
        self.apply_box()
        self.update_positions()
        
    def apply_langevin_dynamics(self):
        for grain in self.StrandA.dnastr + self.StrandB.dnastr:
            # reset velocities
            # grain.velocity = np.array([0.0, 0.0, 0.0])
            # Random thermal force
            random_force = np.random.normal(0, np.sqrt(2 * lamb * kb * temp / dt), size=3)
            # Damping force
            damping_force = -lamb * grain.velocity # with lamb = 1, zeroes previous velocity
            # Total force
            total_force = random_force + damping_force
            grain.update_velocity(total_force, dt)
            
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
            new_trajA.append(grain.position)
        self.trajectoryA.append(new_trajA)
        
        new_trajB = []
        for grain in self.StrandB.dnastr:
            new_trajB.append(grain.position)
        self.trajectoryB.append(new_trajB)
        
        self.energies.append(self.find_energy())
        totpair, selfpair = self.count_tot()
        self.pair_counts.append([totpair, selfpair])
        
        self.endtoends.append(self.endtoend(-1))
    
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
        return totpairs, pairsA - totpairs  
          
    def find_energy(self):
        '''Includes electrostatic and WLC bending energies ONLY'''
        return self.StrandA.eng_elastic() + self.StrandB.eng_elastic() + self.StrandA.eng_elec(self.StrandB)
