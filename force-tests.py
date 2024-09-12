# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 15:14:59 2024

@author: Ralph Holden

Tests the effect of electrostatic force vs fluctuation

Simulates the ~'gradient descent' in electrostatic potential with the Langevin modified Velocity-Verlet algorithm
"""

import matplotlib.pyplot as plt
import numpy as np

import sys
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the target folder
target_dir = os.path.join(script_dir, '../Electrostatics_functions/')

# Add the target directory to sys.path
sys.path.insert(0, os.path.abspath(target_dir))

# Now try to import the module
from Electrostatics_classholder import Electrostatics

correlation_length = 25
grain_mass = 1
kb = 1
temp = 310.15
dt = 0.000001

num_steps = 10000

homology_set = True
elstats = Electrostatics(homol=homology_set)

# Use Kroger with very large gamma (actually coefficient of friction in this case)
# Or not, with normal 0 < gamma < 1

# define parameters for Langevin modified Velocity-Verlet algorithm - Martin Kroger
xi = 2/dt * 0.5 # used instead of gamma for Langevin modified Velocity-Verlet
half_dt = dt/2
applied_friction_coeff = (2 - xi*dt)/(2 + xi*dt)
fluctuation_size = np.sqrt( grain_mass * kb * temp * xi * half_dt ) # takes dt into account. should it be /grain_mass ?
rescaled_position_step = 2*dt / (2 + xi*dt)

print(f''' 
PARAMETERS
    dt         : {dt}
    gamma / xi : {xi}
    half_dt    : {half_dt}
    applied_friction_coeff : {applied_friction_coeff}
    rescaled_position_step : {rescaled_position_step}
      ''')

for Lindex in [1,2,3,4,5]:
    
    Lindex = 'homol '+str(Lindex) if homology_set == True else Lindex

    if True:
        fs = []
        rs = np.linspace(0.2e-8 , 0.6e-8, 1000)
        for r in rs:
            fs += [elstats.find_energy(Lindex,r)]
        plt.figure()
        plt.title(f'Energy vs R, Lindex = {Lindex}')
        plt.plot(rs,fs,label='Electrostatics')
        #plt.ylim(np.min(fs),-np.min(fs))
        plt.legend(loc='best')
        plt.show()

    if True:
        fluctuation_size = np.sqrt(2 * grain_mass * xi * kb * temp / dt)
        fs = []
        rs = np.linspace(0.2e-8 , 0.6e-8, 1000)
        rfs = []
        for r in rs:
            fs += [elstats.force(Lindex,r)]
            rfs += [np.linalg.norm(np.random.normal(0, fluctuation_size, size=3)) *1e-8]
        plt.figure()
        plt.title(f'Force vs R, Lindex = {Lindex}')
        plt.plot(rs,fs,label='Electrostatics')
        #plt.plot(rs,rfs,label=f'Dissipation, xi = {xi}')
        #plt.ylim(np.min(fs),np.max(rfs))
        plt.legend(loc='best')
        plt.show()
        
    if True:
        fluctuation_size = np.sqrt(2 * grain_mass * xi * kb * temp * half_dt)
        vs = []
        rs = np.linspace(0.2e-8 , 0.6e-8, 1000)
        rvs = []
        for r in rs:
            vs += [elstats.force(Lindex,r) * half_dt]
            rvs += [np.linalg.norm(np.random.normal(0, fluctuation_size, size=3)) *1e-8]
        plt.figure()
        plt.title(f'$\delta$ Velocity vs R, Lindex = {Lindex}')
        plt.plot(rs,vs,label='Electrostatics')
        #plt.plot(rs,rvs,label=f'Dissipation, xi = {xi}')
        #plt.ylim(np.min(vs),np.max(rvs))
        plt.legend(loc='best')
        plt.show()
        
    
    
    # # # Langevin modified Velocity-Verlet step - 1D # # #
    
    if False:
        # starting position
        position = [0.25e-8]
        # starting velocity
        velocity = [0]
        # forces
        ext_force = []
    
        for n in range(num_steps):
            # get random fluctuation
            fluctuation_size = np.sqrt(2 * grain_mass * xi * kb * temp * half_dt) 
            fluctuation = (np.random.normal(0, fluctuation_size, size=1) *1e-8 )[0] # scale ??
            
            # calc force
            ext_force.append(elstats.force(Lindex, position[-1]))
            
            # apply force & fluctuation to velocity
            
            velocity.append( velocity[-1] + half_dt * ext_force[-1] )
            velocity[-1] += fluctuation
            
            velocity_firststep_save = velocity[-1]
            
            # apply velocity to position
            position.append(position[-1] + velocity[-1] * rescaled_position_step)
            #position.append(position[-1] + velocity[-1] * dt )
            
            # calc force
            ext_force.append(elstats.force(Lindex, position[-1]))
            
            # apply friction & force & fluctuation to velocity
            velocity[-1] *= applied_friction_coeff
            #velocity[-1] -= velocity[-1] * gamma
            velocity[-1] += half_dt * ext_force[-1]
            velocity[-1] += fluctuation
            
            if n%1000 == 0:
                print(f''' 
                Step {n+1}:
                      
                Start position  {position[-2]}
                Start velocity  {velocity[-2]}
                
                External force  {ext_force[-2]}
                delta velocity  {half_dt*ext_force[-2]}
                Fluctuation     {fluctuation}
                
                New velocity    {velocity_firststep_save}
                
                New position    {position[-1]}
                
                External force  {ext_force[-1]}
                delta velocity  {half_dt*ext_force[-1]}
                Fluctuation     {fluctuation}
                
                New velocity    {velocity[-1]}
                
                
                      ''')
                  
            if not velocity[-1] < 50 and not velocity[-1] > 50:
                print('TERMINATING - lost grains')
                break 
            
        num_steps = len(position)
        plt.figure()
        plt.title(f'Timestep vs Separation, R, Lindex = {Lindex}')
        plt.plot(position,np.linspace(1,num_steps,num_steps),label=f'Dissipation, xi = {xi}')
        plt.ylabel('Step')
        plt.xlabel('R')
        plt.legend(loc='best')
        plt.show()