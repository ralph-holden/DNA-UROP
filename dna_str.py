# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 17:03:37 2024

@author: 44775 Ralph Holden

Nodes are one correllation length of KL theory

Information encoded as string with positions on 'imaginary lattice', instead of using an ising lattice
"""
# # # IMPORTS # # #
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

# # # UNITS # # #
kb = 1
temp = 300

# # # VECTORS # # #
i = np.array([1, 0, 0])
j = np.array([0, 1, 0])
k = np.array([0, 0, 1])
vector_list = [-i,i,-j,j,-k,k]

# # # aux functions # # #
def gen_adj_sites(dna_list: list, index: int) -> list:
    '''Given a segment in a DNA string, outputs all adjacent sites as a list of lists
    Currently, for simple cubic lattice
    '''
    list_out = []
    vector_adj = [i, j, k, i+j, i-j, i+k, i-k, j+k, j-k, i+j+k, i+j-k, i-j-k, i-j+k]
    for vec in vector_adj:
        list_out.append(list(dna_list[index]+vec))
        list_out.append(list(dna_list[index]-vec))
    return list_out

def gen_closeadj_sites(dna_list: list, index: int) -> list:
    '''Given a segment in a DNA string, outputs all adjacent sites as a list of lists
    Currently, for simple cubic lattice
    closest (6) adjacent sites ONLY
    '''
    list_out = []
    vector_adj = [i, j, k]
    for vec in vector_adj:
        list_out.append(list(dna_list[index]+vec))
        list_out.append(list(dna_list[index]-vec))
    return list_out

def count_adj_same(str_s: list, index: int) -> float:
    '''Counts total number of paired segments within the same DNA strand
    for single specified segment only
    considering ONLY CLOSEST adjacent sites
    '''
    count = -2 #does not include required neighbours
    for seg in str_s:
        if list(seg) in gen_closeadj_sites(str_s, index):
            count += 1 
    return count
    
def count_adj_other(str_A: list, index: int, str_B: list):
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
            self.dnastr_A.append(np.array( [ dna_A_start[0], dna_A_start[1]+seg, dna_A_start[2] ] ))
            self.dnastr_B.append(np.array( [ dna_B_start[0], dna_B_start[1]+seg, dna_B_start[2] ] ))
    
    # CHECKS for valid move
    def check_excvol(self, proposed_change: np.array) -> bool:
        '''Outputs boolean for if the proposed change in the metropolis algorithm overlaps with the coordinates of another segment
        True if NO excluded volume errors, False if coordinate overlap
        '''
        for i in range(self.lengths):
            if list(proposed_change) == list(self.dnastr_A[i]):
                return False
            if list(proposed_change) == list(self.dnastr_B[i]):
                return False
            return True
    
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
        else:
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
        
    # for energies of Metropolis
    def eng_elastic(self, str_1: list, str_2: list) -> float:
        '''Energy term for bending of DNA strand from straight
        INCOMPLETE
        '''
        return 0
    
    def eng_elec(self, str_1: list, str_2: list) -> float:
        '''Energy term for electrostatic interactions
        *** IMPORTANT *** must contain condition that if multiple segments in a row are paired, the latter interaction is REPULSIVE
        '''
        return 0
        
    # montecarlo step using metropolis algorithm
    def montecarlostep(self):
        '''Propagating step. Uses a Metropolis algorithm.
        Each time method is called, single step on one DNA strand only
        '''
        # random move from a random particle
        # choose random dna string, segment and vector
        dnastr_rand = np.random.randint(2)
        index_rand = np.random.randint(self.lengths)
        vector_rand = vector_list[np.random.randint(5)]
        
        # apply random change
        dnastr_chosen = [self.dnastr_A,self.dnastr_B][dnastr_rand]
        dnastr_new = dnastr_chosen[:index_rand] + [dnastr_chosen[index_rand]+vector_rand] + dnastr_chosen[index_rand+1:]
        dnastr_other = [self.dnastr_A,self.dnastr_B][dnastr_rand-1]
        
        # test random change against simulation requirements; intact, no overlap, confinement
        if self.check_excvol(dnastr_new[index_rand]) and self.check_strintact(dnastr_new,index_rand) and self.check_inbox(dnastr_new[index_rand]):
            energy_old = self.eng_elastic(self.dnastr_A,self.dnastr_B) + self.eng_elec(self.dnastr_A,self.dnastr_B)
            energy_new = self.eng_elastic(dnastr_new,dnastr_other) + self.eng_elec(dnastr_new, dnastr_other)
            delt_eng = energy_new - energy_old
            # could use the index and its neighbours to calculate the energy change directly
            
            if delt_eng <= 0: # assign proposed string change
                if dnastr_rand == 0:
                    self.dnastr_A[index_rand] = dnastr_new[index_rand]
                elif dnastr_rand == 1:
                    self.dnastr_B[index_rand] = dnastr_new[index_rand]
                    
            elif delt_eng >= 0:
                random_factor = np.random.random()
                boltzmann_factor = np.e**(-1*delt_eng/(kb*temp)) # delt_eng in kb units
                if random_factor < boltzmann_factor: # assign proposed string change
                    if dnastr_rand == 0:
                        self.dnastr_A[index_rand] = dnastr_new[index_rand]
                    elif dnastr_rand == 1:
                        self.dnastr_B[index_rand] = dnastr_new[index_rand]
        
    # functions for data
    def total_adj(self) -> float:
        '''Simply sums total adjacent segments, not including between direct chain connections
        Uses the functions; count_adj_same & count_adj_other
        NOTE: only consideres (6) closest adjacent sites
        '''
        tot_count = 2*2*2 #cancles out -2 from each end segment
        for seg in range(self.lengths):
            tot_count += count_adj_same(self.dnastr_A, seg)/2
            tot_count += count_adj_same(self.dnastr_B, seg)/2
            tot_count += count_adj_other(self.dnastr_A, seg, self.dnastr_B)
        return tot_count
    
    def end_to_end(self) -> Tuple[float, float]:
        return [np.linalg.norm(self.dnastr_A[0]-self.dnastr_A[-1]), np.linalg.norm(self.dnastr_B[0]-self.dnastr_B[-1])]
    
    # # # visualise # # #
    # 2D projection
    def proj_2d(self):
        Ax_points = np.array(self.dnastr_A)[:,0]
        Ay_points = np.array(self.dnastr_A)[:,1]
        Az_points = np.array(self.dnastr_A)[:,2]
        Bx_points = np.array(self.dnastr_B)[:,0]
        By_points = np.array(self.dnastr_B)[:,1]
        Bz_points = np.array(self.dnastr_B)[:,2]
        
        plt.figure()
        plt.subplot(1,2,1)
        plt.title('XY plane')
        plt.plot(Ax_points, Ay_points, marker='o')
        plt.plot(Bx_points, By_points, marker='o')
        plt.xlim(self.xmin,self.xmax)
        plt.ylim(self.ymin,self.ymax)
        plt.grid()
        
        plt.subplot(1,2,2)
        plt.title('XZ plane')
        plt.plot(Ax_points, Az_points, marker='o')
        plt.plot(Bx_points, Bz_points, marker='o')
        plt.xlim(self.xmin,self.xmax)
        plt.ylim(self.zmin,self.zmax)
        plt.grid()
        
        plt.tight_layout(pad=1)
        plt.show()

    
