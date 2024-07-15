# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:06:51 2024

@author: 44775 Ralph Holden

MODEL:
    ball & sticks DNA - like polymer
    balls / particles joined up by straight sticks
    balls / particles correspond to one correlation length as described in Kornyshev-Leiken theory
        as such, for non homologous DNA, only one consecutive correlation length can have attractive charged double helix interactions
   
CODE & SIMULATION:
    Metropolis algorithm (a Monte Carlo method) used to propagate DNA strands
    Additional requirements for a random move are; excluded volume effects, keeping the strand intact, and keeping inside the simulation box (confined DNA, closed simulation)
"""
# # # IMPORTS # # #
import numpy as np
from typing import Tuple
from itertools import combinations

# # # UNITS # # #
kb = 1
temp = 310.15

# PARAMS
lp = 4.5 # persistence length, in correlation length diameter grains of 100 Angstroms
kappab = lp * kb * temp # bending stiffness


# # # aux functions # # #

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
        r = np.linalg.norm([self.x,self.y,self.z])
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
    def __init__(self, position: Vector):
        self.position = position
        self.radius = 0.5
        
    def overlap(self, other) -> Tuple[bool, float]: 
        inter_vector = self.position.arr - other.position.arr
        min_approach = self.radius + other.radius + 0.01 # tolerance to account for rounding errors that could stall the simulation
        #dist = inter_vector.norm()
        dist = np.linalg.norm(inter_vector)
        return dist <= min_approach, dist
    
    def inter_vector(self, other) -> Tuple[bool, float]: 
        intervec = other.position.arr - self.position.arr
        return intervec
    
    def copy(self):
        return Bead(Vector(self.position.x,self.position.y,self.position.z))
    
    
class Strand:
    
    def __init__(self, num_segments: int, start_position: Vector, initial = True, prev_dnastr = None):
        
        self.num_segments = num_segments
        self.start_position = start_position
        
        if initial:
            self.dnastr = [Bead(start_position)]
            for seg in range(num_segments-1):
                self.dnastr.append( Bead( self.dnastr[-1].position + Vector(0,1,0) ) )
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
        for seg in self.dnastr:
            if self.dnastr[index].overlap(seg)[0]:
                count += 1
        return count
    
    def count_adj_other(self, selfindex, other) -> int:
        '''
        For an index, counts number of paired segments with the other DNA strand.
        For single specified segment only.
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
        for seg in self.dnastr:
            if abs(seg.position.x)+0.5 > boxlims.x or abs(seg.position.y)+0.5 > boxlims.y or abs(seg.position.z)+0.5 > boxlims.z:
                return False
        return True 
    
    # for energies
    def condition_interactivity(self, other, fororbac: int, first: bool, seg_index: int, num = 3) -> Tuple[int]:
        '''Defines an interaction as attractive (-1) if it is 'standalone', otherwise repulsive (1) or no interaction (0)
        '''
        if first:
            if self.count_adj_same(seg_index) + self.count_adj_other(seg_index, other) >= num:
                return [-1]
            else:
                return [0]
        if self.count_adj_same(seg_index) + self.count_adj_other(seg_index, other) >= num and abs(self.interactivity[seg_index+fororbac]) == 0:
            return [-1]
        elif self.count_adj_same(seg_index) + self.count_adj_other(seg_index, other) >= num and abs(self.interactivity[seg_index+fororbac]) != 0: 
            return [abs(self.interactivity[seg_index+fororbac])+1] # more repulsive than previous by kbT
        else:
            return [0]
        
    def gen_interactivity(self, other) -> list:
        '''Prioritises sticking to middle (first assigned hence without dependance on +- 1)
        Chooses random index in middle to start on
        NOTE 1: If segment occupies same lattice site as previous (stored length) the second interaction is repulsive (beyond correlation length)
        NOTE 2: As adjacent sites ONLY are counted, stored lengths will not cause issues for interactivity count

        * * * UPDATE REQUIRED: make successive replusions greater in magnitude * * *
        '''
        # starter
        random_start_index = np.random.randint(int(self.lengths/10),int(9*self.lengths/5))
        self.interactivity = self.condition_interactivity(other, 0, True, random_start_index, 3)
        
        # forwards
        for seg_index in range(random_start_index+1,self.num_segments-1): # from index+1 to penultimate
            self.interactivity += self.condition_interactivity(other, -1, False, seg_index, 3)
            
        # end (from forwards)
        self.interactivity += self.condition_interactivity(other, -1, False, -1, 2)
        
        # backwards
        for seg_index in np.linspace(random_start_index-1, 1, random_start_index-1): # from index-1 to second index
            self.interactivity = self.condition_interactivity(other, +1, False, seg_index, 3) + self.interactivity
            
        # end (from backwards)
        self.interactivity = self.condition_interactivity(other, +1, False, 0, 2) + self.interactivity
    
    def eng_elec(self):
        return np.sum(self.interactivity)

    def find_angle(self, seg_index):
        # Points p1, p2, and p3 are arrays or tuples of the form [x, y]
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
    
    def entropic_bend(self):
        return 0.0
    
    def free_energy(self):
        return self.eng_elec() + self.eng_elastic() + temp*self.entropic_bend()
    
    def statistics(self):
        pass
    
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
        
        rand_theta = rand1*np.pi/360/2 * [-1,1][np.random.randint(2)]
        rand_phi = rand2*np.pi/360  /2 * [-1,1][np.random.randint(2)]
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
    
    
    def propose_change_both(self, seg_indexA: int, seg_indexB: int, forward = True):
        
        prop_Strand = self.copy()
        
        rand_thetaA = np.random.random()*np.pi/720 * [-1,1][np.random.randint(2)]
        rand_phiA = np.random.random()*np.pi/720   * [-1,1][np.random.randint(2)]
        rand_thetaB = np.random.random()*np.pi/720 * [-1,1][np.random.randint(2)]
        rand_phiB = np.random.random()*np.pi/720   * [-1,1][np.random.randint(2)]
        # at most a 0.25 degree bend allowed
        # shift every subsequent bead, NOTE: not applicable for final bead
        if forward:
            for nextseg in range(seg_indexA+1, self.num_segments): # bends down one direction of chain
                prop_Strand.dnastr[nextseg].position = prop_Strand.calc_arc(seg_indexA, nextseg, rand_thetaA, rand_phiA)
            for nextseg in range(seg_indexB+1, self.num_segments): # bends down one direction of chain
                prop_Strand.dnastr[nextseg].position = prop_Strand.calc_arc(seg_indexB, nextseg, rand_thetaB, rand_phiB)
        elif not forward:
            for nextseg in range(seg_indexA-1, -1, -1):
                prop_Strand.dnastr[nextseg].position = prop_Strand.calc_arc(seg_indexA, nextseg, rand_thetaA, rand_phiA)
            for nextseg in range(seg_indexB-1, -1, -1):
                prop_Strand.dnastr[nextseg].position = prop_Strand.calc_arc(seg_indexB, nextseg, rand_thetaB, rand_phiB)
        return prop_Strand
    
    
    def propose_change_both_whole(self):
        '''Doesnt work! Dont use'''
        # random initial segment in middle 3/5 of DNA strand
        random_start_indexA = np.random.randint(int(self.num_segments/10),int(9*self.num_segments/10))
        random_start_indexB = np.random.randint(int(self.num_segments/10),int(9*self.num_segments/10))
        # in current model, do not need to propose a change to the starting segmentrandom_start_indexA = np.random.randint(int(self.num_segments/10),int(9*self.num_segments/10))
        # make copy for first time, then after, update that
        # going forwards, updating entire rest of strand each time
        prop_StrandA = self.propose_change(random_start_indexA, forward = True) # first bend
        prop_StrandB = self.propose_change(random_start_indexB, forward = True) # first bend
        for seg_index in range(random_start_indexA+1, self.num_segments-1): # again, final bead cannot bend a further
            prop_StrandA = prop_StrandA.propose_change(seg_index, forward = True)
        for seg_index in range(random_start_indexB+1, self.num_segments-1): # again, final bead cannot bend a further
            prop_StrandB = prop_StrandB.propose_change(seg_index, forward = True)
        for seg_index in range(random_start_indexA+1, 0, -1): # again, final bead cannot bend a further
            prop_StrandA = prop_StrandA.propose_change(seg_index, forward = False)
        for seg_index in range(random_start_indexB+1, 0, -1): # again, final bead cannot bend a further
            prop_StrandB = prop_StrandB.propose_change(seg_index, forward = False)
        return prop_StrandA, prop_StrandB
    
class Simulation:
    
    nsteps = 1
    mctime = 0.0
    
    def __init__(self, boxlims: Vector, StrandA: Strand, StrandB: Strand):
        self.boxlims = boxlims # boxlims a Vector
        
        self.StrandA = StrandA
        self.StrandB = StrandB
        
        self.free_energy = self.StrandA.free_energy() + self.StrandB.free_energy()
    
        self.trajectoryA = []
        self.trajectoryB = []
        self.pair_count = []
        self.fe_traj = []
        self.save_trajectory()
        
        

        
        
    def montecarlostep_trial(self):
        #prop_StrandA, prop_StrandB = self.StrandA.propose_change_both_whole()
        prop_StrandA = self.StrandA.propose_change_whole(random_start_index = np.random.randint(int(self.StrandA.num_segments/10),int(9*self.StrandA.num_segments/10)))
        prop_StrandB = self.StrandB.propose_change_whole(random_start_index = np.random.randint(int(self.StrandB.num_segments/10),int(9*self.StrandB.num_segments/10)))
                
        # find valid configuration, need to wait for entire strand to change before excvol and inbox can fairly be applied
        while not prop_StrandA.check_excvol(prop_StrandB) or not prop_StrandA.check_inbox(self.boxlims) or not prop_StrandB.check_inbox(self.boxlims) or not prop_StrandA.check_strintact_whole() or  not prop_StrandB.check_strintact_whole():
           #prop_StrandA, prop_StrandB = self.StrandA.propose_change_both_whole()
           prop_StrandA = self.StrandA.propose_change_whole(np.random.randint(int(self.StrandA.num_segments/10),int(9*self.StrandA.num_segments/10)))
           prop_StrandB = self.StrandB.propose_change_whole(np.random.randint(int(self.StrandB.num_segments/10),int(9*self.StrandB.num_segments/10)))
        
        # calculate deltaE 
        prev_energy = self.free_energy 
        prop_energy = prop_StrandA.free_energy() + prop_StrandB.free_energy()
        deltaE = prop_energy - prev_energy
    
        if deltaE <= 0: # assign new string change, which has already 'passed' conditions from proposal 
            self.StrandA, self.StrandB = prop_StrandA, prop_StrandB
            self.free_energy = prop_energy
            self.mctime += 0.0 # assign energy, strings and trajectories
    
        elif deltaE >= 0:
            random_factor = np.random.random()
            boltzmann_factor = np.e**(-1*deltaE/(1)) # delt_eng in kb units
            if random_factor < boltzmann_factor: # assign new string change
                self.StrandA, self.StrandB = prop_StrandA, prop_StrandB
                self.free_energy = prop_energy 
                self.mctime += 0.0 # assign energy, strings and trajectories
                    
        self.nsteps += 1
        self.save_trajectory()
        
    def montecarlostep(self):
        prop_StrandA = self.StrandA.propose_change_whole()
        prop_StrandB = self.StrandB.propose_change_whole()
                
        # test for valid configuration
        if prop_StrandA.check_excvol(prop_StrandB) and prop_StrandA.check_inbox(self.boxlims) and prop_StrandB.check_inbox(self.boxlims) and prop_StrandA.check_strintact_whole() and prop_StrandB.check_strintact_whole():
        
            # calculate deltaE 
            prev_energy = self.free_energy 
            prop_energy = prop_StrandA.free_energy() + prop_StrandB.free_energy()
            deltaE = prop_energy - prev_energy
        
            if deltaE <= 0: # assign new string change, which has already 'passed' conditions from proposal 
                self.StrandA, self.StrandB = prop_StrandA, prop_StrandB
                #self.trajectoryA.append(self.StrandA.dnastr)
                #self.trajectoryB.append(self.StrandB.dnastr)
                self.save_trajectory()
                self.free_energy = prop_energy
                self.mctime += 0.0 # assign energy, strings and trajectories
        
            elif deltaE >= 0:
                random_factor = np.random.random()
                boltzmann_factor = np.e**(-1*deltaE/(temp)) # delt_eng in kb units
                if random_factor < boltzmann_factor: # assign new string change
                    self.StrandA, self.StrandB = prop_StrandA, prop_StrandB
                    #self.trajectoryA.append(self.StrandA.dnastr)
                    #self.trajectoryB.append(self.StrandB.dnastr)
                    self.save_trajectory()
                    self.free_energy = prop_energy 
                    self.mctime += 0.0 # assign energy, strings and trajectories
                        
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
        
        self.fe_traj.append(self.free_energy)
    
    def endtoend(self, tindex):
        endtoendA = np.linalg.norm(self.trajectoryA[tindex][0] - self.trajectoryA[tindex][-1])
        endtoendB = np.linalg.norm(self.trajectoryB[tindex][0] - self.trajectoryB[tindex][-1])
        return endtoendA, endtoendB
    
    def count_tot(self, other):
        '''Discounts immediate neighbours from pairs'''
        pass
        #self.paired_count = 0
        #if self.count_adj_same(0) >= 2:
        #    self.paired_count += 1
        #if self.count_adj_same(seg_index) + self.count_adj_other(seg_index, other) >= num and abs(self.interactivity[seg_index+fororbac]) == 0:
        #    return [-1]
        #elif self.count_adj_same(seg_index) + self.count_adj_other(seg_index, other) >= num and abs(self.interactivity[seg_index+fororbac]) != 0: 
        #    return [abs(self.interactivity[seg_index+fororbac])+1] # more repulsive than previous by kbT
        #else:
        #    return [0]
    
    def statistics(self):
        pass
