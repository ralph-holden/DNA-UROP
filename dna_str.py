# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 17:03:37 2024

@author: 44775 Ralph Holden

MODEL:
    ball & sticks DNA - like polymer
    balls / particles occupy sites on a lattice, joined up by straight sticks
    balls / particles correspond to one correlation length as described in Kornyshev-Leiken theory
        as such, for non homologous DNA, only one consecutive correlation length can have attractive charged double helix interactions
    ball & stick model has translational freedom, can move & bend, but chain links are unable to stretch or twist
        assuming then, that these stretches and twists occur instantaneously with respect to a DNA configuration, so can be accounted for simply as a perturbation of the energy
    
CODE & SIMULATION:
    Information coded as list of 3D coordinates of DNA nodes on a simple cubic lattice (for now)
    Metropolis algorithm (a Monte Carlo method) used to propagate DNA strands
    Additional requirements for a random move are; excluded volume effects, keeping the strand intact, and keeping inside the simulation box (confined DNA, closed simulation)
"""
# # # IMPORTS # # #
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

# # # UNITS # # #
kb = 1
temp = 310.15

# # # VECTORS # # #
i = np.array([1, 0, 0])
j = np.array([0, 1, 0])
k = np.array([0, 0, 1])

vector_list = [] # this is used for moving the segments
for n in [i+j, i-j, i+k, i-k, j+k, j-k]:
    vector_list.append(n/(2**0.5))
    vector_list.append(-n/(2**0.5)) 

# # # aux functions # # #
def dot(vec1: np.array, vec2: np.array):
    '''Calculate the dot product between two 3D vectors'''
    return vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]

def norm(vec):
    '''Calculate the norm of the 2D vector'''
    return (vec[0]**2 + vec[1]**2 + vec[2]**2)**0.5

def angle(vec1, vec2):
    return np.arccos( dot(vec1,vec2) / ( norm(vec1)*norm(vec2) ) )

# # # aux functions # # #

def gen_adj_sites(dna_list: list, index: int) -> list:
    '''Given a segment in a DNA string, outputs all adjacent sites as a list of lists
    For 12 closest sites on an FCC lattice
    '''
    list_out = []
    for vec in vector_list:
        list_out.append(list(dna_list[index]+vec))
        list_out.append(list(dna_list[index]-vec))
    return list_out

def gen_closeadj_sites(dna_list: list, index: int) -> list:
    '''
    * * * OUTDATED * * *
    Given a segment in a DNA string, outputs all adjacent sites as a list of lists
    Currently, for simple cubic lattice
    closest (6) adjacent sites ONLY
    '''
    list_out = []
    vector_adj = [i, j, k]
    for vec in vector_adj:
        list_out.append(list(dna_list[index]+vec))
        list_out.append(list(dna_list[index]-vec))
    return list_out

def count_adj_same(str_s: list, index: int) -> int:
    '''Counts total number of paired segments within the same DNA strand
    for single specified segment only
    considering ALL adjacent sites, otherwise count does not work
    '''
    count = 0 #does not include required neighbours
    for seg in str_s:
        if list(seg) in gen_adj_sites(str_s, index):
            count += 1 
    return count
    
def count_adj_other(str_A: list, index: int, str_B: list) -> int:
    '''Counts the total number of paired segments with the other DNA strand
    for single specified segment only
    considering ONLY CLOSEST adjacent sites
    '''
    count = 0 
    for seg in str_B:
        if list(seg) in gen_closeadj_sites(str_A, index):
            count += 1 
    return count        
    

# # # Monte Carlo Model Class # # #
class dna_string_model:
    
    n_steps = 0
    mctime = 0
    
    def __init__(self, n_x: int, n_y: int, n_z: int, dna_A_start: int, dna_B_start: int, dna_lengths: int):
        # lattice / 'box'
        self.xmin = -n_x
        self.xmax =  n_x
        self.ymin = -n_y
        self.ymax =  n_y
        self.zmin = -n_z
        self.zmax =  n_z
        # initialise straight DNA chains as strings with each unit as XYZ coordinates
        self.lengths = dna_lengths
        self.dnastr_A = []
        self.dnastr_B = []
        for seg in range(self.lengths):
            self.dnastr_A.append(np.array( [ dna_A_start[0], dna_A_start[1]+seg*vector_list[0], dna_A_start[2] ] ))
            self.dnastr_B.append(np.array( [ dna_B_start[0], dna_B_start[1]+seg*vector_list[0], dna_B_start[2] ] ))
        self.trajectories = [[self.dnastr_A, self.dnastr_B]]
        self.interactivity_A = []
        self.interactivity_B = []
    
    # CHECKS for valid move
    def check_excvol(self, proposed_change: np.array) -> bool:
        '''Outputs boolean for if the proposed change in the metropolis algorithm overlaps with the coordinates of another segment
        True if NO excluded volume errors, False if coordinate overlap
        NOTE: in first generation montecarlostep(), the segment HAS to move and therefore coincide with any of the previous / OLD strand
        '''
        for i in range(self.lengths):
            if list(proposed_change) == list(self.dnastr_A[i]):
                return False
            if list(proposed_change) == list(self.dnastr_B[i]):
                return False
        return True # only AFTER checking ALL DNA strand sites for a False / failure 
        
    def check_excvol_gen2(self, dnastr_A_prop: list, dnastr_B_prop: list, index_prop: int) -> bool:
        '''Outputs boolean for if the proposed change in the metropolis algorithm overlaps with the coordinates of another segment
        True if NO excluded volume errors, False if coordinate overlap
        NOTE: must allow DNA segments to stay in the same place, but need to avoid flagging up excluded volume error
        '''
        proposed_change_A = dnastr_A_prop[index_prop]
        proposed_change_B = dnastr_B_prop[index_prop]
        for i in range(self.lengths):
            if index_prop not in [i-1,i,i+1]: # ignore proposals that stayed in the same place, or neighbour segments
                if list(proposed_change_A) == list(dnastr_A_prop[i]) or list(proposed_change_B) == list(dnastr_B_prop[i]):
                    return False # excluded volume error with self, ignoring the neigbour segment for pseudo-lattice-elasticity
            if list(proposed_change_A) == list(dnastr_B_prop[i]) or list(proposed_change_B) == list(dnastr_A_prop[i]):
                return False # excluded volume error with other strand
            if list(proposed_change_A) == list(proposed_change_B):
                return False # special case, both DNA segments trying to move into the same place
        return True # only AFTER checking ALL DNA strand sites for a False / failure
    
    def check_strintact(self, dnastr: list, index: int) -> bool:
        ''' Outputs boolean for if the segment of the proposed change is still adjacent (including diagnols of technically greater length) to its neighbouring segments
        True if DNA chain still intact, False if connection broken
        '''
        adjacent_sites = gen_adj_sites(dnastr, index)
        if index == 0: # start segment
            if list(dnastr[index+1]) in adjacent_sites:
                return True
        elif index == self.lengths - 1: # end segment
            if list(dnastr[index-1]) in adjacent_sites:
                return True
        else: # any middle segment
            if list(dnastr[index-1]) in adjacent_sites and list(dnastr[index+1]) in adjacent_sites:
                return True
        return False
    
    def check_inbox(self, proposed_change: np.array) -> bool:
        ''' Outputs boolean for if the segment of the proposed chain is still within the simulation box,
        as per confinement
        True if within box, False if outside of box'''
        for i,dir in enumerate([[self.xmin,self.xmax],[self.ymin,self.ymax],[self.zmin,self.zmax]]):
            if proposed_change[i] < dir[0] or proposed_change[i] > dir[1]:
                return False
        return True
    
    # for number adjacent DNA strands in separate
    def numb_adj(self) -> float:
        '''Needs to be for each set of paired DNA'''
        #could use the sum of adjacent sites
        
    def condition_interactivity(self, AorB: str, fororbac: int, first: bool, seg: int, num: int) -> Tuple[int]:
        '''Defines an interaction as attractive (1) if it is 'standalone', otherwise repulsive (-1) or no interaction (0)
        '''
        if AorB == 'A':
            dnastr_s = self.dnastr_A
            dnastr_o = self.dnastr_B
            interactivity = self.interactivity_A
        elif AorB == 'B':
            dnastr_s = self.dnastr_B
            dnastr_o = self.dnastr_A
            interactivity = self.interactivity_B
            
        if first:
            if count_adj_same(dnastr_s, seg) + count_adj_other(dnastr_s, seg, dnastr_o) >= num:
                return [1]
            else:
                return [0]
            
        if count_adj_same(dnastr_s, seg) + count_adj_other(dnastr_s, seg, dnastr_o) >= num and abs(interactivity[seg+fororbac]) != 1:
            return [1]
        elif count_adj_same(dnastr_s, seg) + count_adj_other(dnastr_s, seg, dnastr_o) >= num and abs(interactivity[seg+fororbac]) == 1: 
            return [-1]
        else:
            return [0]
        
    def gen_interactivity(self) -> list:
        '''Prioritises sticking to middle (first assigned hence without dependance on +- 1)
        Chooses random index in middle to start on'''
        # starter
        random_start_index = np.random.randint(int(self.lengths/5),int(4*self.lengths/5))
        self.interactivity_A = self.condition_interactivity('A', 0, True, random_start_index, 3)
        self.interactivity_B = self.condition_interactivity('B', 0, True, random_start_index, 3)
        
        # forwards
        for seg in range(random_start_index+1,self.lengths-1): # from index+1 to penultimate
            self.interactivity_A += self.condition_interactivity('A', -1, False, seg, 3)
            self.interactivity_B += self.condition_interactivity('B', -1, False, seg, 3)
            
        # end (from forwards)
        self.interactivity_A += self.condition_interactivity('A', -1, False, seg, 2)
        self.interactivity_B += self.condition_interactivity('B', -1, False, seg, 2)
        
        # backwards
        for seg in np.linspace(random_start_index-1, 1, random_start_index-1): # from index-1 to second index
            self.interactivity_A = self.condition_interactivity('A', +1, False, seg, 3) + self.interactivity_A
            self.interactivity_B = self.condition_interactivity('B', +1, False, seg, 3) + self.interactivity_B
            
        # end (from backwards)
        self.interactivity_A = self.condition_interactivity('A', +1, False, seg, 2) + self.interactivity_A
        self.interactivity_B = self.condition_interactivity('B', +1, False, seg, 2) + self.interactivity_B
        
    def propose_change(self, dnastr_A_new, dnastr_B_new, index_rand):
    
        vector_list_zero = vector_list + [np.array([0.0])]
        vector_A_rand = vector_list_zero[np.random.randint(13)] # length of list is 13, indexes 0-12
        vector_B_rand = vector_list_zero[np.random.randint(13)]

        dnastr_A_prop = dnastr_A_new[:index_rand] + [dnastr_A_new[index_rand]+vector_A_rand] + dnastr_A_new[index_rand+1:]

        dnastr_B_prop = dnastr_B_new[:index_rand] + [dnastr_B_new[index_rand]+vector_B_rand] + dnastr_B_new[index_rand+1:]

        return dnastr_A_prop, dnastr_B_prop
        
    # for energies of Metropolis
    def eng_elastic_pb(self, dnastr, seg_index: int) -> float:
        vec1 = dnastr[seg_index-1] - dnastr[seg_index]
        vec2 = dnastr[seg_index+1] - dnastr[seg_index]
        return 1000 * (1 - np.cos(angle(vec1, vec2))**2) # completely arbitrary! 
        # * * * UPDATE REQUIRED: need to use averaged angles * * *
        # * * * UPDATE REQUIRED: need to use elastic rod model * * *
    
    def eng_elastic(self, str_1: list, str_2: list) -> float:
        '''Energy term for bending of DNA strand from straight
        INCOMPLETE
        '''
        energy = 0
        for seg_index in range(1,self.lengths-1):
            energy += self.eng_elastic_pb(self.dnastr_A, seg_index)
            energy += self.eng_elastic_pb(self.dnastr_B, seg_index)
        return energy

    def entropic_bend(self):
        return 0
        # * * * UPDATE REQUIRED: need to incorporate into energy terms to give the FREE ENERGY * * *
    
    def eng_elec(self, str_1: list, str_2: list) -> float:
        '''Energy term for electrostatic interactions
        *** IMPORTANT *** must contain condition that if multiple segments in a row are paired, the latter interaction is REPULSIVE
        '''
        factor = -0.1 # completely arbitrary, but from interactivity needs to be negative
        return np.sum(factor*np.array(self.interactivity_A) + factor*np.array(self.interactivity_B))
    
    def eng(self) -> float:
        '''Returns total energy of current configuration'''
        return self.eng_elastic(self.dnastr_A,self.dnastr_B) + self.eng_elec(self.dnastr_A,self.dnastr_B) + self.entropic_bend()*temp
        
    # montecarlo step using metropolis algorithm
    def montecarlostep(self):
        '''Propagating step. Uses a Metropolis algorithm.
        Each time method is called, single step on one DNA strand only
        '''
        # random move from a random particle
        # choose random dna string, segment and vector
        dnastr_rand = np.random.randint(2)
        index_rand = np.random.randint(self.lengths)
        vector_rand = vector_list[np.random.randint(13)]
        
        # apply random change
        dnastr_chosen = [self.dnastr_A,self.dnastr_B][dnastr_rand]
        dnastr_new = dnastr_chosen[:index_rand] + [dnastr_chosen[index_rand]+vector_rand] + dnastr_chosen[index_rand+1:]
        dnastr_other = [self.dnastr_A,self.dnastr_B][dnastr_rand-1]
        
        # test random change against simulation requirements; intact, no overlap, confinement
        if self.check_excvol(dnastr_new[index_rand]) and self.check_strintact(dnastr_new,index_rand) and self.check_inbox(dnastr_new[index_rand]):
            energy_old = self.eng() # * * * UPDATE REQUIRED: can remove to save computational cost * * *
            energy_new = self.eng_elastic(dnastr_new,dnastr_other) + self.eng_elec(dnastr_new, dnastr_other) + self.entropic_bend()*temp
            delt_eng = energy_new - energy_old
            # could use the index and its neighbours to calculate the energy change directly
            
            if delt_eng <= 0: # assign proposed string change
                if dnastr_rand == 0:
                    self.dnastr_A[index_rand] = dnastr_new[index_rand]
                elif dnastr_rand == 1:
                    self.dnastr_B[index_rand] = dnastr_new[index_rand]
                    
                self.trajectories.append([self.dnastr_A,self.dnastr_B])
                    
            elif delt_eng >= 0:
                random_factor = np.random.random()
                boltzmann_factor = np.e**(-1*delt_eng/(kb*temp)) # delt_eng in kb units
                if random_factor < boltzmann_factor: # assign proposed string change
                    if dnastr_rand == 0:
                        self.dnastr_A[index_rand] = dnastr_new[index_rand]
                    elif dnastr_rand == 1:
                        self.dnastr_B[index_rand] = dnastr_new[index_rand]
                        
                    self.trajectories.append([self.dnastr_A,self.dnastr_B])
                    
        self.n_steps += 1
        
    def montecarlostep_gen2(self):
        '''Propagating step. Uses a Metropolis algorithm.
        Each time method is called, entire system updated! New configuration then accepted or rejected.
        '''
        # random moves
        # choose random dna index, in same fashion as interactivity, segment and vector
        random_start_index = np.random.randint(int(self.lengths/5),int(4*self.lengths/5))
        
        # propose a single change to the DNA strands until it meets the conditions, then initalise build of NEW DNA strands
        dnastr_A_prop, dnastr_B_prop = self.propose_change(self.dnastr_A, self.dnastr_B, random_start_index) # propose change to random_start_index
        # test random change against simulation requirements; intact, no overlap, confinement
        while not self.check_excvol_gen2(dnastr_A_prop,dnastr_B_prop,random_start_index) and not self.check_strintact(dnastr_A_prop,random_start_index) and not self.check_strintact(dnastr_A_prop,random_start_index) and not self.check_inbox(dnastr_A_prop[random_start_index]) and not self.check_inbox(dnastr_B_prop[random_start_index]):
            dnastr_A_prop, dnastr_B_prop = self.propose_change(self.dnastr_A, self.dnastr_B, random_start_index) # propose new change until satisfied
        dnastr_A_new, dnastr_B_new = dnastr_A_prop, dnastr_B_prop # proposed change works with simulation rules
        
        # forwards
        for seg_index in range(random_start_index+1, self.lengths): 
            dnastr_A_prop, dnastr_B_prop = self.propose_change(dnastr_A_new, dnastr_B_new, seg_index) # propose change to seg_index
            # test random change against simulation requirements; intact, no overlap, confinement
            while not self.check_excvol_gen2(dnastr_A_prop,dnastr_B_prop,seg_index) and not self.check_strintact(dnastr_A_prop,seg_index) and not self.check_strintact(dnastr_A_prop,seg_index) and not self.check_inbox(dnastr_A_prop[seg_index]) and not self.check_inbox(dnastr_B_prop[random_start_index]):
                    dnastr_A_prop, dnastr_B_prop = self.propose_change(dnastr_A_new, dnastr_B_new, seg_index) # propose new change until satisfied
            dnastr_A_new, dnastr_B_new = dnastr_A_prop, dnastr_B_prop # proposed change works with simulation rules
            
        # backwards
        for seg_index in np.linspace(random_start_index-1, 0, random_start_index):
            seg_index = int(seg_index)
            dnastr_A_new, dnastr_B_new = self.propose_change(dnastr_A_new, dnastr_B_new, seg_index) # propose change
            # test random change against simulation requirements; intact, no overlap, confinement
            while not self.check_excvol_gen2(dnastr_A_prop,dnastr_B_prop,seg_index) and not self.check_strintact(dnastr_A_prop,seg_index) and not self.check_strintact(dnastr_A_prop,seg_index) and not self.check_inbox(dnastr_A_prop[seg_index]) and not self.check_inbox(dnastr_B_prop[random_start_index]):
                    dnastr_A_prop, dnastr_B_prop = self.propose_change(dnastr_A_new, dnastr_B_new, seg_index) # propose new change until satisfied     

        energy_old = self.eng() # * * * UPDATE REQUIRED: can remove to save computational cost * * *
        energy_new = self.eng_elastic(dnastr_new,dnastr_other) + self.eng_elec(dnastr_new, dnastr_other) + self.entropic_bend()*temp
        delt_eng = energy_new - energy_old
            # could use the index and its neighbours to calculate the energy change directly

        if delt_eng <= 0: # assign new string change, which has already 'passed' conditions from proposal 
            self.dnastr_A, self.dnastr_B = dnastr_A_new, dnastr_B_new
            self.trajectories.append([self.dnastr_A,self.dnastr_B])

        elif delt_eng >= 0:
            random_factor = np.random.random()
            boltzmann_factor = np.e**(-1*delt_eng/(kb*temp)) # delt_eng in kb units
            if random_factor < boltzmann_factor: # assign new string change
                self.dnastr_A, self.dnastr_B = dnastr_A_new, dnastr_B_new
                self.trajectories.append([self.dnastr_A,self.dnastr_B])

        self.n_steps += 1
        self.mctime += 0 # * * * UPDATE REQUIRED: introduce dynamic Monte Carlo timestep * * *
        
    # functions for data
    def statistics_inst(self) -> Tuple[float, Tuple[int,int,int,int], Tuple[float,float], int]:
        """Returns the averaged values of energy"""
        return [self.eng(), [self.total_adj()], [self.end_to_end()], self.n_steps]
    
    ### NEED TO MAKE FUNCTION / CHANGE ARCHITECTURE TO HAVE RUNNING AVERAGES OF DATA, NOT JUST FINAL
    
    def total_adj(self) -> Tuple[int, int, int]:
        '''Simply sums total adjacent segments, not including between direct chain connections
        Uses the functions; count_adj_same & count_adj_other
        NOTE: only consideres (6) closest adjacent sites
        '''
        same_A_count = 2*2 # cancels out -2 from each end segment
        same_B_count = 2*2
        other_count = 0
        tot_count = 0 
        for seg in range(self.lengths):
            same_A_count += count_adj_same(self.dnastr_A, seg)/2
            same_B_count += count_adj_same(self.dnastr_B, seg)/2
            other_count += count_adj_other(self.dnastr_A, seg, self.dnastr_B)
        tot_count = same_A_count + same_B_count + other_count 
            
        return tot_count, same_A_count, same_B_count, other_count
    
    def end_to_end(self) -> Tuple[float, float]:
        return np.linalg.norm(self.dnastr_A[0]-self.dnastr_A[-1]), np.linalg.norm(self.dnastr_B[0]-self.dnastr_B[-1])
    
    # # # visualise # # #
    # 2D projection
    def proj_2d(self, fullbox = True):
        Ax_points = np.array(self.dnastr_A)[:,0]
        Ay_points = np.array(self.dnastr_A)[:,1]
        Az_points = np.array(self.dnastr_A)[:,2]
        Bx_points = np.array(self.dnastr_B)[:,0]
        By_points = np.array(self.dnastr_B)[:,1]
        Bz_points = np.array(self.dnastr_B)[:,2]
        
        plt.figure()
        plt.subplot(1,2,1)
        plt.title('XY plane')
        plt.plot(Ax_points, Ay_points, marker='o', markersize=0.5)
        plt.plot(Bx_points, By_points, marker='o', markersize=0.5)
        if fullbox:
            plt.xlim(self.xmin,self.xmax)
            plt.ylim(self.ymin,self.ymax)
        plt.grid()
        
        plt.subplot(1,2,2)
        plt.title('XZ plane')
        plt.plot(Ax_points, Az_points, marker='o', markersize=0.5)
        plt.plot(Bx_points, Bz_points, marker='o', markersize=0.5)
        if fullbox:
            plt.xlim(self.xmin,self.xmax)
            plt.ylim(self.zmin,self.zmax)
        plt.grid()
        
        plt.tight_layout(pad=1)
        plt.show()
    
    def proj_3d(self):
        Ax_points = np.array(self.dnastr_A)[:,0]
        Ay_points = np.array(self.dnastr_A)[:,1]
        Az_points = np.array(self.dnastr_A)[:,2]
        Bx_points = np.array(self.dnastr_B)[:,0]
        By_points = np.array(self.dnastr_B)[:,1]
        Bz_points = np.array(self.dnastr_B)[:,2]
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        
        x= Ax_points
        y= Ay_points
        z= Az_points 
        ax.plot3D(x, y, z, 'red')
        
        x= Bx_points
        y= By_points
        z= Bz_points 
        ax.plot3D(x, y, z, 'blue')
        
        ax.set_xlabel('x ($l_c$)')
        ax.set_ylabel('y ($l_c$)')
        ax.set_zlabel('z ($l_c$)')
        plt.show()
