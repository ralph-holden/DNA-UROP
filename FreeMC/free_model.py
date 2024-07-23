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

# PARAMS
lp = 4.5 # persistence length, in coherence length diameter grains of 100 Angstroms
kappab = lp * kb * temp # bending stiffness


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
    
    
class Strand:
    '''
    Each dsDNA strand involved in the simulation
    Worm-Like-Chain bead angles
    Kornyshev-Leiken electrostatic interactions
    '''
    
    def __init__(self, num_segments: int, start_position: Vector, initial = True, prev_dnastr = None):
        
        self.num_segments = num_segments
        self.start_position = start_position
        
        if initial:
            self.dnastr = [Bead(start_position)]
            for seg in range(num_segments-1):
                self.dnastr.append( Bead( self.dnastr[-1].position + Vector(0,0.2,0) ) )
        elif not initial:
            self.dnastr = [prev_dnastr[0].copy()]
            for seg in prev_dnastr[1:]:
                self.dnastr.append( Bead( Vector(seg.position.x,seg.position.y,seg.position.z)))
        
        self.interactivity = []
        
        self.fe = 0
        
    def copy(self):
        '''Create a new object which is a copy of the current.'''
        #Strandnew = Strand(self.num_segments, self.start_position)
        # Strandnew.dnastr = self.dnastr # make sure new DNA strand is up to date
        return (Strand(self.num_segments, self.start_position, initial = False, prev_dnastr = self.dnastr)) #Strandnew
    
    def count_adj_same(self, index: int) -> int:
        '''
        For an index, counts number of paired segments within the same DNA strand.
        Considering ALL adjacent sites, otherwise count does not work.
        '''
        count = 0 # does not include required neighbours or oneself
        for b in self.dnastr:
            if self.dnastr[index].overlap(b)[0]:
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
    
    # CHECKS for valid move
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
        
    
    def propose_change(self, seg_index: int, rand1, rand2, forward = True):
        
        prop_Strand = self.copy()
        
        rand_theta = rand1*np.pi/360/10 * [-1,1][np.random.randint(2)]
        rand_phi = rand2*np.pi/360  /10 * [-1,1][np.random.randint(2)]
        # shift every subsequent bead, NOTE: not applicable for final bead
        if forward:
            for nextseg in range(seg_index+1, self.num_segments): # bends down one direction of chain
                prop_Strand.dnastr[nextseg].position = prop_Strand.calc_arc(seg_index, nextseg, rand_theta, rand_phi)
        elif not forward:
            for nextseg in range(seg_index-1, -1, -1):
                prop_Strand.dnastr[nextseg].position = prop_Strand.calc_arc(seg_index, nextseg, rand_theta, rand_phi)
        return prop_Strand
    
    
    def propose_change_whole(self, random_start_index: int):
        # random initial segment in middle 3/5 of DNA strand
        #random_start_index = np.random.randint(int(self.num_segments/10),int(9*self.num_segments/10))
        # in current model, do not need to propose a change to the starting segment
        # make copy for first time, then after, update that
        # going forwards, updating entire rest of strand each time
        prop_Strand = self.propose_change(random_start_index, np.random.random(), np.random.random(), forward = True) # first bend
        for seg_index in range(random_start_index+1, self.num_segments-1): # again, final bead cannot bend a further
            prop_Strand = prop_Strand.propose_change(seg_index, np.random.random(), np.random.random(), forward = True)
        for seg_index in range(random_start_index+1, 0, -1): # again, final bead cannot bend a further
            prop_Strand = prop_Strand.propose_change(seg_index, np.random.random(), np.random.random(), forward = False)
        return prop_Strand
    
    

    
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
        self.save_trajectory()
        
        
    def montecarlostep(self):
        #prop_StrandA, prop_StrandB = self.StrandA.propose_change_both_whole()
        prop_StrandA = self.StrandA.propose_change_whole(random_start_index = np.random.randint(int(self.StrandA.num_segments/10),int(9*self.StrandA.num_segments/10)))
        prop_StrandB = self.StrandB.propose_change_whole(random_start_index = np.random.randint(int(self.StrandB.num_segments/10),int(9*self.StrandB.num_segments/10)))
                
        # find valid configuration, need to wait for entire strand to change before excvol and inbox can fairly be applied
        while not prop_StrandA.check_excvol(prop_StrandB) or not prop_StrandA.check_inbox(self.boxlims) or not prop_StrandB.check_inbox(self.boxlims) or not prop_StrandA.check_strintact_whole() or  not prop_StrandB.check_strintact_whole():
           #prop_StrandA, prop_StrandB = self.StrandA.propose_change_both_whole()
           prop_StrandA = self.StrandA.propose_change_whole(np.random.randint(int(self.StrandA.num_segments/10),int(9*self.StrandA.num_segments/10)))
           prop_StrandB = self.StrandB.propose_change_whole(np.random.randint(int(self.StrandB.num_segments/10),int(9*self.StrandB.num_segments/10)))
        
        # calculate deltaE 
        prev_energy = self.energy 
        prop_energy = prop_StrandA.eng_elastic() + prop_StrandB.eng_elastic() + prop_StrandA.eng_elec(prop_StrandB)
        deltaE = prop_energy - prev_energy
    
        if deltaE <= 0: # assign new string change, which has already 'passed' conditions from proposal 
            self.StrandA, self.StrandB = prop_StrandA, prop_StrandB
            self.energy = prop_energy
            self.mctime += 0.0 # assign energy, strings and trajectories
    
        elif deltaE >= 0:
            random_factor = np.random.random()
            boltzmann_factor = np.e**(-1*deltaE/(temp)) # deltaE in kb units
            if random_factor < boltzmann_factor: # assign new string change
                self.StrandA, self.StrandB = prop_StrandA, prop_StrandB
                self.energy = prop_energy 
                self.mctime += 0.0 # assign energy, strings and trajectories
                     
        self.save_trajectory()
        self.nsteps += 1
            
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
    
    def statistics(self):
        pass
