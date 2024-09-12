# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 15:14:59 2024

@author: Ralph Holden

1 Dimensional version of Langevin simulation

Applies bending & electrostatic forces, and fluctuations
Only allows motion along x axis

Reduced effects of entropy in 1D (less degrees of freedom)
As such, MORE likely to aggregate than, more physical, 3D case
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

import logging
from datetime import datetime

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(script_dir, '../Electrostatics_functions/')
sys.path.insert(0, os.path.abspath(target_dir))
from Electrostatics_classholder import Electrostatics

# # # PARAMETERS # # #
num_steps = 10000
coherence_lengths = 2
nsegs = 5 * coherence_lengths
xsep  = 0.25e-8
xsep += 0.2e-8 # surface to surface
dt = 0.0000005
gamma = 0.5
xi = 2/dt * gamma

homology_set = True
elstats = Electrostatics(homol=homology_set)

# outputs
frame_hop = 100
log_update = 1000

# Worm Like Chain Bending
kb = 1
temp = 310.15
lp = 5e-8 # persistence length, in coherence length diameter grains of 100 Angstroms
kappab = lp * kb * temp # bending stiffness
s = 0.4e-8 # standard distance through chain separated by one Grain
k_bend = kappab/(2*s) # Bending stiffness constant

correlation_length = 5
grain_mass = 1
kb = 1
temp = 310.15

# define parameters for Langevin modified Velocity-Verlet algorithm - Martin Kroger
half_dt = dt/2
applied_friction_coeff = (2 - xi*dt)/(2 + xi*dt)
fluctuation_size = np.sqrt( grain_mass * kb * temp * xi * half_dt ) # takes dt into account. should it be /grain_mass ?
rescaled_position_step = 2*dt / (2 + xi*dt)

for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
log_filename = datetime.now().strftime('./Data_outputs/LOG_%Y%m%d_%H%M%S.log')
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')
logging.info('Simulation started')
logging.info(f'''Simulation parameters:
    dt         : {dt}
    gamma      : {gamma}
    xi         : {xi}
    half_dt    : {half_dt}
    applied_friction_coeff : {applied_friction_coeff}
    rescaled_position_step : {rescaled_position_step}
             ''')

# # # Langevin modified Velocity-Verlet step - 1D # # #
xpositionA = np.ones( nsegs ) * (+xsep/2)
xpositionB = np.ones( nsegs ) * (-xsep/2)
yposition = np.arange(0, nsegs*0.2e-8, 0.2e-8)
# trajectories
xpositionA_traj = []
xpositionB_traj = []
for i in range( len(xpositionA) ):
    xpositionA_traj.append( [xpositionA[i]] )
    xpositionB_traj.append( [xpositionB[i]] ) 
# starting velocity
velocityA = np.zeros(nsegs) # only in x direction
velocityB = np.zeros(nsegs)
# forces
#ext_forceA = ([],) * nsegs
#ext_forceB = ([],) * nsegs

# data
energies = []

L_list_traj = []

# calculate angles
def f_wlc(xposition):
    torque_list = np.zeros(nsegs)
    for i in range(1, len(xposition) - 1):
        # Vectors between adjacent grains
        p1 = np.array([ xposition[i-1], yposition[i-1] ])
        p2 = np.array([ xposition[i]  , yposition[i]   ])
        p3 = np.array([ xposition[i+1], yposition[i+1] ])
        r1 = p1 - p2
        r2 = p3 - p2
        r1_norm = np.linalg.norm(r1)
        r2_norm = np.linalg.norm(r2)
        # Cosine of the angle between r1 and r2
        theta = np.arccos(np.clip(np.dot(r1, r2) / (r1_norm * r2_norm), -1.0, 1.0))
        torque_magnitude = -k_bend * (theta - np.pi) 

        torque_direction = (p1[0]+p3[0])/2 - p2[0]
        torque_direction /= np.linalg.norm(torque_direction) if np.linalg.norm(torque_direction) != 0 else 1
        torque = torque_magnitude * torque_direction

        #torque_list[i-1] -= torque
        torque_list[i]   += torque # which to augment!?
        #torque_list[i+1] -= torque

    return torque_list 

# calculate electrostatics
def find_L():
    # find minimum L
    Lindex_min = np.argmin( xpositionA - xpositionB ) if np.std(xpositionA - xpositionB) > 2e-12 else int(nsegs/2)
    L_list = [1]
    for i in range(Lindex_min-1, -1, -1):
        L_list = [L_list[0]+1] + L_list if not homology_set else ['homol 0']+L_list
    for i in range(Lindex_min+1, len(xpositionA)):
        L_list += [L_list[-1]+1] if not homology_set else ['homol 0']
    return L_list

def f_elstat(L_list):
    force_list = []
    for i, Lindex  in enumerate(L_list):
        force_list += [elstats.force(Lindex, abs(xpositionA[i] - xpositionB[i])-0.2e-8)]
    return np.array(force_list)

def eng_elstat(L_list):
    energy = 0.0
    for n in range(nsegs):
        g_R = (xpositionA - xpositionB)[n] - 0.2e-8
        g_L = L_list[n]
        ishomol = type(g_L) == str
        if not ishomol:
            energy +=  elstats.find_energy(g_L, g_R) - elstats.find_energy(g_L-1, g_R) # remove 'built-up' energy over L w/ different R
        elif ishomol:
            energy += elstats.find_energy(g_L, g_R) # energy is per unit length
    return energy / (kb*300) # give energy in kbT units
    
def eng_elastic(xposition) -> float:
    energy = 0
    for i in range(1,nsegs-1):
        p1 = np.array([ xposition[i-1], yposition[i-1] ])
        p2 = np.array([ xposition[i]  , yposition[i]   ])
        p3 = np.array([ xposition[i+1], yposition[i+1] ])
        r1 = p1 - p2
        r2 = p3 - p2
        r1_norm = np.linalg.norm(r1)
        r2_norm = np.linalg.norm(r2)
        # Cosine of the angle between r1 and r2
        theta = np.arccos(np.clip(np.dot(r1, r2) / (r1_norm * r2_norm), -1.0, 1.0))
        energy += 1/2 * k_bend *  (theta-np.pi)**2
    return energy / (kb*300) # give energy in kbT units

for n in range(num_steps):
    
    # get random fluctuation
    fluctuation_size = np.sqrt(2 * grain_mass * xi * kb * temp * half_dt)
    fluctuationA , fluctuationB = [] , []
    for i in range(int(np.ceil(nsegs/correlation_length))):
        fluctuationA += [(np.random.normal(0, fluctuation_size, size=1) *1e-8)[0]] * correlation_length
        fluctuationB += [(np.random.normal(0, fluctuation_size, size=1) *1e-8)[0]] * correlation_length
    fluctuationA, fluctuationB = fluctuationA[:nsegs] , fluctuationB[:nsegs] 
    
    # calc force
    electrostatic_force = f_elstat(find_L())
    ext_forceA = +electrostatic_force + f_wlc(xpositionA)
    ext_forceB = -electrostatic_force + f_wlc(xpositionB)
    ext_force_firststep_save = ext_forceA
    
    # apply force & fluctuation to velocity
    velocityA += half_dt * ext_forceA
    velocityA += fluctuationA
    velocityB += half_dt * ext_forceB
    velocityB += fluctuationB
    
    velocity_firststep_save = velocityA
    
    separation_prestep_save = xpositionA-xpositionB
    # apply velocity to position
    xpositionA += velocityA * rescaled_position_step
    xpositionB += velocityB * rescaled_position_step
    for i in range( len(xpositionA) ):
        xpositionA_traj[i].append( xpositionA[i] )
        xpositionB_traj[i].append( xpositionB[i] )  
    
    # calc force
    electrostatic_force = f_elstat(find_L())
    ext_forceA = +electrostatic_force + f_wlc(xpositionA)
    ext_forceB = -electrostatic_force + f_wlc(xpositionB)
    ext_force_firststep_save = ext_forceA
    
    # apply friction & force & fluctuation to velocity
    velocityA *= applied_friction_coeff
    velocityA += half_dt * ext_forceA
    velocityA += fluctuationA
    velocityB *= applied_friction_coeff
    velocityB += half_dt * ext_forceB
    velocityB += fluctuationB
    
    L_list_traj.append(find_L())
    energies += [eng_elstat(find_L()) + eng_elastic(xpositionA) + eng_elastic(xpositionB)]
    
    if n%log_update == 0:
        print(f''' 
        Step {n+1}:
              
        Start separation {np.mean(separation_prestep_save)-0.2e-8}
        Start xvelocity  {np.mean(velocityA-velocityB)}
        
        External force   {np.mean(ext_force_firststep_save)}
        delta velocity   {np.mean(half_dt*ext_force_firststep_save)}
        Fluctuation A    {np.mean(fluctuationA)}
        
        New xvelocity    {np.mean(velocity_firststep_save)}
        
        New position A   {np.mean(xpositionA)}
        New position B   {np.mean(xpositionB)}
        New separation   {np.mean(xpositionA-xpositionB)-0.2e-8}
        
        External force   {np.mean(ext_forceA)}
        delta velocity   {np.mean(half_dt*ext_forceA)}
        Fluctuation A    {np.mean(fluctuationA)}
        
        New xvelocity    {np.mean(velocityA)}
        
        New energy       {energies[-1]}
              ''')
              
        logging.info(f'''Step {n} : DATA:
        Mean Separation  {np.mean(xpositionA-xpositionB)-0.2e-8}
        STD  Separation  {np.std(xpositionA-xpositionB)}
        Energy       {energies[-1]}
                      ''')
              
    if not velocityA[-1] < 50 and not velocityA[-1] > 50 or not velocityB[-1] < 50 and not velocityB[-1] > 50:
        print('TERMINATING - lost grains')
        break 
    
print(f'''Simulation completed
      Mean Separation  {np.mean(xpositionA-xpositionB)-0.2e-8}
      STD  Separation  {np.std(xpositionA-xpositionB)}
      Energy       {energies[-1]}
      
      ''')
logging.info(f'''Simulation completed
      Mean Separation  {np.mean(xpositionA-xpositionB)-0.2e-8}
      STD  Separation  {np.std(xpositionA-xpositionB)}
      Energy       {energies[-1]}
      
      ''')
      
 
num_steps = len(xpositionA_traj[0]) # reset number of steps

if True:
    plt.figure(figsize=[16,10])
    plt.title('Internal Energy')
    plt.plot(range(1,num_steps),energies)
    plt.xlabel('Timestep')
    plt.ylabel('Energy (kbT)')
    plt.grid(linestyle=':')
    plt.savefig('./Data_outputs/1D_energies.png')
    plt.show()
    
if True:
    plt.figure(figsize=[8,5])
    plt.title('Leading Pair')
    plt.plot(range(1,num_steps),[np.argmin(t) for t in L_list_traj])
    plt.xlabel('Timestep')
    plt.ylabel('Index of Leading Pair')
    plt.grid(linestyle=':')
    #plt.savefig('./Data_outputs/1D_leading_pair.png')
    plt.show()

if True:
    plt.figure(figsize=[16,10])
    plt.title('Timestep vs Separation, R')
    for i in range(nsegs):
        plt.subplot(int(nsegs/2),2,i+1)
        plt.plot(np.array(xpositionA_traj[i])-np.array(xpositionB_traj[i])-0.2e-8,np.linspace(1,num_steps,num_steps))
        plt.ylabel('Step')
        plt.xlabel('R')
    plt.tight_layout(pad=1)
    plt.savefig('./Data_outputs/1D_pair_separations.png')
    plt.show()

# # # ANIMATION # # #
if True:
    selected_frames = range(0,num_steps,frame_hop)
    num_frames = len(selected_frames)
    
    fig = plt.figure(figsize=[16,10])
    ax = fig.add_subplot()
    
    data1_init, data2_init = [] , []
    for n in range(nsegs):    
        data1_init.append( xpositionA_traj[n][0] ) , data2_init.append( xpositionB_traj[n][0] )
    
    # Initial data
    x1 = np.array(data1_init)
    y1 = yposition
    x2 = np.array(data2_init)
    y2 = yposition
    
    # set limits
    #ax.set_xlim(np.min(xpositionA),np.max(xpositionB))
    
    # Initial line plots
    line1, = ax.plot(x1-0.1e-8, y1, color='b', marker='.', markersize=1, label='DNA strand A')
    line2, = ax.plot(x1+0.1e-8, y1, color='b', marker='.', markersize=1)
    line3, = ax.plot(x2-0.1e-8, y2, color='r', marker='.', markersize=1, label='DNA strand B')
    line4, = ax.plot(x2+0.1e-8, y2, color='r', marker='.', markersize=1)
    
    # Create a text label for the frame number, initially set to the first frame
    frame_text = ax.text(0.05, 0.95, f"Frame: {selected_frames[0]}", transform=ax.transAxes)
    
    ax.legend()
    
    def update(frame):
        # Fetch data for the current frame directly
        x1 = np.array([xpositionA_traj[n][frame] for n in range(nsegs)])
        x2 = np.array([xpositionB_traj[n][frame] for n in range(nsegs)])
        
        # Update line plots with new data
        line1.set_data(x1-0.1e-8, y1)
        line2.set_data(x1+0.1e-8, y1)
        line3.set_data(x2-0.1e-8, y1)
        line4.set_data(x2+0.1e-8, y2)
        
        
        # Update the frame number text
        frame_text.set_text(f"Frame: {selected_frames[frame]}")
        
        return line1, line2, line3, line4, frame_text

    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)
    
    # Save the animation as an MP4 file (uncomment to save)
    ani.save('./Data_outputs/1d_line_animation.gif', writer=PillowWriter(fps=20))
    
    # Show the plot
    plt.show()