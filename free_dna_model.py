# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 17:03:37 2024

@author: 44775 Ralph Holden

MODEL:
    ball & sticks DNA - like polymer
    balls / particles occupy sites on a lattice, joined up by straight sticks
    balls / particles correspond to one correlation length as described in Kornyshev-Leiken theory
        as such, for non homologous DNA, only one consecutive correlation length can have attractive charged double helix interactions
    ball & stick model has translational freedom, can move & bend, is also pseudo-elastic, allowing motion 'down chain'
        assuming then, that these twists occur instantaneously with respect to a DNA configuration, so can be accounted for simply as a perturbation of the energy
    
CODE & SIMULATION:
    Metropolis algorithm (a Monte Carlo method) used to propagate DNA strands
    Additional requirements for a random move are; excluded volume effects, keeping the strand intact, and keeping inside the simulation box (confined DNA, closed simulation)
"""
# # # IMPORTS # # #
import numpy as np
from typing import Tuple
#import matplotlib.pyplot as plt
from itertools import combinations

# # # UNITS # # #
kb = 1
temp = 310.15

# # # vector class # # #
# for maths
class Vector():
    '''3D vectors'''

    def __init__(self,i1,i2,i3):
        '''Initialise vectors with x and y and z coordinates'''
        self.x = i1
        self.y = i2
        self.z = i3

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


# # # bead class # # #
class Bead: 
    def __init__(self, position: Vector):
        self.position = position
        self.radius = 1
        
    def overlap(self, other) -> Tuple[bool, float]: 
        inter_vector = self.position - other.position
        min_approach = self.radius + other.radius + 0.01 # tolerance to account for rounding errors that could stall the simulation
        dist = inter_vector.norm()
        return dist <= min_approach, dist
    
    
class Strand:
    def __init__(self, num_segments: int, start_position: Vector):
        
        self.num_segments = num_segments
        self.start_position = start_position
        
        self.dnastr = [Bead(start_position)]
        for seg in range(num_segments-1):
            self.dnastr.append( Bead( start_position+Vector(0,2,0) ) )
        
        self.interactivity = []
        
        self.fe = 0
        
    def copy(self):
        '''Create a new object which is a copy of the current.'''
        Strandnew = Strand(self.num_segments, self.start_position)
        Strandnew.dnastr = self.dnastr # make sure new DNA strand is up to date
        return Strandnew
    
    def count_adj_same(self, index: int) -> int:
        '''
        For an index, counts number of paired segments within the same DNA strand.
        Considering ALL adjacent sites, otherwise count does not work.
        '''
        count = 0 # does not include required neighbours or oneself
        for seg in self.dnastr:
            if self.dnastr[index].overlap(seg)[0]:
                count += 1
        return count
    
    def count_adj_other(self, selfindex, other) -> int:
        '''
        For an index, counts number of paired segments with the other DNA strand.
        For single specified segment only.
        Considering ONLY CLOSEST adjacent sites.
        '''
        count = 0 
        for seg in other.dnastr:
            if self.dnastr[selfindex].overlap(seg)[0]:
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
            if segA.overlap(segB)[1] < segA.radius: # when overlap is TOO great, defined as the centre of one overlapping with other, so still allows some overlap to register interactions
                return False
        return True
    
    def check_strintact(self, index: int) -> bool:
        ''' 
        Outputs boolean for if the segment of the proposed change is still adjacent
        True if DNA chain still intact (bead contact or overlap), False if connection broken
        '''
        if index == 0: # start segment
            return self.dnastr[index].overlap(index+1)[0]
        elif index == self.num_segments - 1: # end segment
            return self.dnastr[index].overlap(index-1)[0]
        elif index > 0 and index < self.num_segments - 1: # any middle segment
            return self.dnastr[index].overlap(index-1)[0] and self.dnastr[index].overlap(index+1)[0]
        
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
        for seg in self.dnastr:
            if abs(seg.position.x)+0.5 > boxlims.x or abs(seg.position.y)+0.5 > boxlims.y or abs(seg.position.z)+0.5 > boxlims.z:
                return False
        return True 
    
    # for energies
    def interactivity_condition(self):
        pass
    
    def interactivity(self):
        pass
    
    def eng_elec(self):
        return -0.0
    
    def eng_elastic(self):
        return -0.0
    
    def entropic_bend(self):
        return 0.0
    
    def free_energy(self):
        return self.eng_elec() + self.eng_elastic() + temp*self.entropic_bend()
    
    def statistics(self):
        pass
    
    # for MC step
    def calc_arc(self, selfindex: int, otherindex: int, thi: float, theta: float) -> Vector:
        '''
        Gives the displacement of the bead for an MC bend
        Theta and thi must be in radians
        '''
        dist = (self.dnastr[selfindex].position - self.dnastr[otherindex].position).norm()
        return self.dnastr[otherindex].position + dist*Vector(np.cos(thi)*np.cos(theta),np.cos(thi)*np.sin(theta),np.sin(thi))
        
    def propose_change(self, seg_index: int, forward = True):
        
        prop_Strand = self.copy()
        
        rand_thi = np.random.random()*np.pi/6 # at most a 30 degree bend allowed
        rand_theta = np.random.random()*np.pi*2 # all meridians allowed
        # shift every subsequent bead, NOTE: not applicable for final bead
        if forward:
            for nextseg in range(seg_index+1, self.num_segments-1): # bends down one direction of chain
                prop_Strand.dnastr[nextseg].position = prop_Strand.calc_arc(seg_index, nextseg, rand_thi, rand_theta)
        elif not forward:
            for nextseg in range(seg_index-1, 0, -1):
                prop_Strand.dnastr[nextseg].position = prop_Strand.calc_arc(seg_index, nextseg, rand_thi, rand_theta)
        return prop_Strand
    
    def propose_change_whole(self):
        # random initial segment in middle 3/5 of DNA strand
        random_start_index = np.random.randint(int(self.num_segments/5),int(4*self.num_segments/5))
        # in current model, do not need to propose a change to the starting segment
        # make copy for first time, then after, update that
        # going forwards, updating entire rest of strand each time
        prop_Strand = self.propose_change(random_start_index, forward = True) # first bend
        for seg_index in range(random_start_index+1, self.num_segments-1): # again, final bead cannot bend a further
            prop_Strand = prop_Strand.propose_change(seg_index, forward = True)
        for seg_index in range(random_start_index+1, self.num_segments-1): # again, final bead cannot bend a further
            prop_Strand = prop_Strand.propose_change(seg_index, forward = True)
        return prop_Strand
        
    
class Simulation:
    
    nsteps = 0
    mctime = 0.0
    
    def __init__(self, boxlims: Vector, StrandA: Strand, StrandB: Strand):
        self.boxlims = boxlims # boxlims a Vector
        
        self.StrandA = StrandA
        self.StrandB = StrandB
        
        self.Sim_free_energy = self.StrandA.free_energy() + self.StrandB.free_energy()
    
        self.trajectoryA = []
        self.trajectoryB = []
        self.save_trajectory()
        
        
    def save_trajectory(self):
        
        for seg in self.StrandA.dnastr:
            self.trajectoryA.append(np.array([seg.position.x,seg.position.y,seg.position.z]))
        
        for seg in self.StrandB.dnastr:
            self.trajectoryB.append(np.array([seg.position.x,seg.position.y,seg.position.z]))
        
        
    def montecarlostep(self):
        prop_StrandA = self.StrandA.propose_change_whole()
        prop_StrandB = self.StrandB.propose_change_whole()
                
        # find valid configuration, need to wait for entire strand to change before excvol and inbox can fairly be applied
        if not prop_StrandA.check_excvol(prop_StrandB) or not prop_StrandA.inbox(self.boxlims) or not prop_StrandB.inbox(self.boxlims) or not prop_StrandA.check_strintact_whole() or  not prop_StrandB.check_strintact_whole():
        #   prop_StrandA = self.StrandA.propose_change_whole()
        #   prop_StrandB = self.StrandB.propose_change_whole() # try again
        
        # calculate deltaE 
            prev_energy = self.Sim_free_energy 
            prop_energy = prop_StrandA.free_energy() + prop_StrandB.free_energy()
            deltaE = prop_energy - prev_energy
    
            if deltaE <= 0: # assign new string change, which has already 'passed' conditions from proposal 
                self.StrandA, self.StrandB = prop_StrandA, prop_StrandB
                #self.trajectoryA.append(self.StrandA.dnastr)
                #self.trajectoryB.append(self.StrandB.dnastr)
                self.save_trajectory()
                self.Sim_free_energy = prop_energy
                self.mctime += 0.0 # assign energy, strings and trajectories
    
            elif deltaE >= 0:
                random_factor = np.random.random()
                boltzmann_factor = np.e**(-1*deltaE/(1)) # delt_eng in kb units
                if random_factor < boltzmann_factor: # assign new string change
                    self.StrandA, self.StrandB = prop_StrandA, prop_StrandB
                    #self.trajectoryA.append(self.StrandA.dnastr)
                    #self.trajectoryB.append(self.StrandB.dnastr)
                    self.save_trajectory()
                    self.Sim_free_energy = prop_energy 
                    self.mctime += 0.0 # assign energy, strings and trajectories
                    
            self.nsteps += 1