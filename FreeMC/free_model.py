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
from scipy import signal
from typing import Tuple
from itertools import combinations

# # # UNITS # # #
kb = 1
temp = 310.15

# # # PARAMETERS # # #
# Worm Like Chain Bending
lp = 4.5 # persistence length, in coherence length diameter grains of 100 Angstroms
# NOTE: lp for specific temperature, expect to decrease w/ temperature increase ?
kappab = lp * kb * temp # bending stiffness
s = 0.4 # standard distance through chain separated by one Grain
k_bend = kappab/s # Bending stiffness constant


# # # aux functions # # #
x = np.linspace(0,1000,int(1000/0.2)+1)
nonhomolfunc = np.concatenate((-x[x <= 1.0],(x-2)[x > 1])) #zero@start, -1 kbT @ 1 lc, +1 kbT @ 2lc, & so on
#homolrecfunc = 

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

# function to generate DNA strand
def create_strand(num_segments, xstart, ystart, zstart):
    dnastr = []
    dnastr.append(Bead(Vector(xstart, ystart, zstart)))
    for seg in range(num_segments - 1):
        new_position = dnastr[-1].position + Vector(0, 0.2, 0)
        dnastr.append(Bead(Vector(new_position.x, new_position.y, new_position.z)))
    return Strand(num_segments, dnastr)
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
        
        self.interactivity = []
        
        self.fe = 0
        
    def copy(self):
        '''Create a new object which is a copy of the current.'''
        #Strandnew = Strand(self.num_segments, self.start_position)
        # Strandnew.dnastr = self.dnastr # make sure new DNA strand is up to date
        newdnastr = [] #List.empty_list(BeadType)
        for i in range(self.num_segments):
            seg = self.dnastr[i]
            newdnastr.append( Bead( Vector(seg.position.x,seg.position.y,seg.position.z)))
        return (Strand(self.num_segments, newdnastr)) 
    
    def count_adj_same(self, index: int) -> int:
        '''
        For an index, counts number of paired segments within the same DNA strand.
        Considering ALL adjacent sites, otherwise count does not work.
        '''
        count = 0 # will NOT count self and neighbours
        for bi in range(self.num_segments):
            if abs(bi-index) > 1:
                count += 1 if self.dnastr[index].overlap(self.dnastr[bi])[0] else 0
        return count
    
    def count_adj_other(self, selfindex, other) -> int:
        '''
        For an index, counts number of paired segments with the other DNA strand.
        For single specified segment only.
        '''
        count = 0 
        for b in other.dnastr:
            count += 1 if self.dnastr[selfindex].overlap(b)[0] else 0
        return count
    
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
    
    # CHECKS for valid move
    def check_count_increase(self, other, strand1, strand2) -> bool:
        '''
        Checks that the TOTAL number of pairs increases in a Monte Carlo move
        Incorporated into the montecarlostep when 'catch = True'
        '''
        return self.count_all(strand1, strand2)[0] >= self.count_all(self, other)[0]
    
    def check_excvol(self, other) -> bool:
        '''
        Outputs boolean for if the proposed change in the metropolis algorithm overlaps with the coordinates of another segment
        True if NO excluded volume errors, False if coordinate overlap
        NOTE: in first generation montecarlostep(), the segment HAS to move and therefore coincide with any of the previous / OLD strand
        '''
        for segA, segB in combinations(self.dnastr+other.dnastr,2):
            if segA.overlap(segB)[1] <= segA.radius: # when overlap is TOO great, defined as the centre of one overlapping with other, so still allows some overlap to register interactions
                return False
        return True
    
    def check_strintact(self, index: int) -> bool:
        ''' 
        Outputs boolean for if the segment of the proposed change is still adjacent
        True if DNA chain still intact (bead contact or overlap), False if connection broken
        '''
        if index == 0: # start segment
            return self.dnastr[index].overlap(self.dnastr[index+1])[0]
        elif index == self.num_segments - 1: # end segment
            return self.dnastr[index].overlap(self.dnastr[index-1])[0]
        elif index > 0 and index < self.num_segments - 1: # any middle segment
            return self.dnastr[index].overlap(self.dnastr[index-1])[0] and self.dnastr[index].overlap(self.dnastr[index+1])[0]
        
    def check_strintact_whole(self):
        for seg_index in range(self.num_segments):
            if self.check_strintact(seg_index) == False:
                return False
        return True
    
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
    
    # for energies
    def condition_interactivity(self, other, fororbac: int, first: bool, seg_index: int, num = 0) -> Tuple[int]:
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
        self.interactivity  = self.condition_interactivity(other, 0, True, 0, 0) # starter
        other.interactivity = other.condition_interactivity(self, 0, True, 0, 0)
        for seg_index in range(1,self.num_segments-1): # from index+1 to penultimate
            self.interactivity  += self.condition_interactivity(other, -1, False, seg_index, 0) # forward
            other.interactivity += other.condition_interactivity(self, -1, False, seg_index, 0)
        self.interactivity  += self.condition_interactivity(other, -1, False, -1, 0) # end
        other.interactivity += other.condition_interactivity(self, -1, False, -1, 0)
    
    def eng_elec_old(self, other):
        energy, eng_bit = 0, 0
        self.gen_interactivity(other)
        for i in self.interactivity: # interactivity gives INDEX of segment on non homologous interaction function
            eng_bit = nonhomolfunc[i] if i != 0 else eng_bit # updates the energy of the paired sequence, overwrites previous
            energy += eng_bit if i == 0 else 0 # when paired sequence ends submit energy of part, if not paired OR sequence 'unfinished' submits 0
            eng_bit = 0 if i == 0 else eng_bit # resets eng_bit at the end of non interacting part
        energy += eng_bit # for case that final segment is paired
        other.gen_interactivity(self)
        eng_bit = 0
        for i in other.interactivity: 
            eng_bit = nonhomolfunc[i] if i != 0 else eng_bit 
            energy += eng_bit if i == 0 else 0 
            eng_bit = 0 if i == 0 else eng_bit 
        energy += eng_bit 
        return energy
    
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
        
        rand_theta = np.random.random()*np.pi/360/10 * [-1,1][np.random.randint(2)]
        rand_phi =   np.random.random()*np.pi/360/10 * [-1,1][np.random.randint(2)]
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
        
        self.energy = self.StrandA.eng_elastic() + self.StrandB.eng_elastic() + self.StrandA.eng_elec(self.StrandB)
    
        self.trajectoryA = []
        self.trajectoryB = []
        self.pair_count = []
        self.eng_traj = []
        self.centremass = []
        self.save_trajectory()
        
        
    def montecarlostep(self):
        #prop_StrandA, prop_StrandB = self.StrandA.propose_change_both_whole()
        prop_StrandA = self.StrandA.propose_change_whole()
        prop_StrandB = self.StrandB.propose_change_whole()
                
        # find valid configuration, need to wait for entire strand to change before excvol and inbox can fairly be applied
        prop_StrandA, prop_StrandB = self.retry(prop_StrandA, prop_StrandB, catch = True)
        
        # calculate deltaE 
        prev_energy = self.energy 
        prop_energy = prop_StrandA.eng_elastic() + prop_StrandB.eng_elastic() + prop_StrandA.eng_elec(prop_StrandB)
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
            while not prop_StrandA.check_excvol(prop_StrandB) or not prop_StrandA.check_inbox(self.boxlims) or not prop_StrandB.check_inbox(self.boxlims) or not prop_StrandB.check_strintact_whole(): #or not prop_StrandA.check_strintact_whole()
               prop_StrandA = self.StrandA.propose_change_whole()
               prop_StrandB = self.StrandB.propose_change_whole()
        if catch:
            while not prop_StrandA.check_excvol(prop_StrandB) or not prop_StrandA.check_inbox(self.boxlims) or not prop_StrandB.check_inbox(self.boxlims) or not prop_StrandB.check_strintact_whole() or not self.StrandA.check_count_increase(self.StrandB, prop_StrandA, prop_StrandB): #or not prop_StrandA.check_strintact_whole()
               prop_StrandA = self.StrandA.propose_change_whole()
               prop_StrandB = self.StrandB.propose_change_whole()
        return prop_StrandA, prop_StrandB
            
    # for data analysis
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
        totpair, selfpair = self.count_tot()
        self.pair_count.append([totpair, selfpair])
        self.centremass.append([self.StrandA.find_centremass(),self.StrandB.find_centremass()])
    
    def endtoend(self, tindex):
        endtoendA = np.linalg.norm(self.trajectoryA[tindex][0] - self.trajectoryA[tindex][-1]) + 0.2 # size of beads themselves
        endtoendB = np.linalg.norm(self.trajectoryB[tindex][0] - self.trajectoryB[tindex][-1]) + 0.2
        return endtoendA, endtoendB
    
    def count_tot(self):
        '''Discounts immediate neighbours from pairs
        Must be run only after gen_interactivity'''
        comparearray = np.zeros(len(self.StrandA.interactivity))
        pairsA = np.sum(np.array(self.StrandA.interactivity)!=comparearray)
        pairsB = np.sum(np.array(self.StrandB.interactivity)!=comparearray)
        totpairs = int((pairsA+pairsB)/2)
        return totpairs, pairsA - totpairs
    
    def statistics(self):
        pass
