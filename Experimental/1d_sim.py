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

# # # PARAMETERS # # #
num_steps = 50000
nsegs = 4
xsep  = 0.25e-8
dt = 0.0000001
xi = 2/dt * 0.5 # used instead of gamma for Langevin modified Velocity-Verlet

homology_set = False
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
k_bend = kappab/s # Bending stiffness constant

correlation_length = 25
grain_mass = 1
kb = 1
temp = 310.15

# define parameters for Langevin modified Velocity-Verlet algorithm - Martin Kroger
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

        torque_list[i-1] -= torque
        #torque_list[i]   += torque # which to augment!?
        torque_list[i+1] -= torque

    return torque_list 

# calculate electrostatics
def find_L():
    # find minimum L
    Lindex_min = np.argmin( xpositionA - xpositionB )
    L_list = [1]
    for i in range(Lindex_min-1, -1, -1):
        L_list = [L_list[0]+1] + L_list
    for i in range(Lindex_min+1, len(xpositionA)):
        L_list += [L_list[-1]+1]
    return L_list

def f_elstat():
    L_list = find_L()
    force_list = []
    for i, Lindex  in enumerate(L_list):
        Lindex = 'homol '+str(Lindex) if homology_set==True else Lindex
        force_list += [elstats.force(Lindex, abs(xpositionA[i] - xpositionB[i]))]
    return np.array(force_list)

for n in range(num_steps):
    
    # get random fluctuation
    fluctuation_size = np.sqrt(2 * grain_mass * xi * kb * temp * half_dt) 
    fluctuationA = (np.random.normal(0, fluctuation_size, size=1) *1e-8)[0]
    fluctuationB = (np.random.normal(0, fluctuation_size, size=1) *1e-8)[0]
    
    # calc force
    ext_forceA = +f_elstat() + f_wlc(xpositionA)
    ext_forceB = -f_elstat() + f_wlc(xpositionB)
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
    ext_forceA = +f_elstat() + f_wlc(xpositionA)
    ext_forceB = -f_elstat() + f_wlc(xpositionB)
    ext_force_firststep_save = ext_forceA
    
    # apply friction & force & fluctuation to velocity
    velocityA *= applied_friction_coeff
    velocityA += half_dt * ext_forceA
    velocityA += fluctuationA
    velocityB *= applied_friction_coeff
    velocityB += half_dt * ext_forceB
    velocityB += fluctuationB
    
    L_list_traj.append(find_L())
    
    if n%log_update == 0:
        print(f''' 
        Step {n+1}:
              
        Start separation {np.mean(separation_prestep_save)}
        Start xvelocity  {np.mean(velocityA-velocityB)}
        
        External force   {np.mean(ext_force_firststep_save)}
        delta velocity   {np.mean(half_dt*ext_force_firststep_save)}
        Fluctuation A    {np.mean(fluctuationA)}
        
        New xvelocity    {np.mean(velocity_firststep_save)}
        
        New position A   {np.mean(xpositionA)}
        New position B   {np.mean(xpositionB)}
        New separation   {np.mean(xpositionA-xpositionB)}
        
        External force   {np.mean(ext_forceA)}
        delta velocity   {np.mean(half_dt*ext_forceA)}
        Fluctuation A    {np.mean(fluctuationA)}
        
        New xvelocity    {np.mean(velocityA)}
        
        
              ''')
              
    if not velocityA[-1] < 50 and not velocityA[-1] > 50 or not velocityB[-1] < 50 and not velocityB[-1] > 50:
        print('TERMINATING - lost grains')
        break 
    
print(f'''Simulation completed
      New position A   {np.mean(xpositionA)}
      New position B   {np.mean(xpositionB)}
      New separation   {np.mean(xpositionA-xpositionB)}
      
      ''')
 
num_steps = len(xpositionA_traj[0]) # reset number of steps

if True:
    plt.figure(figsize=[16,10])
    plt.title('Timestep vs Separation, R')
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.plot(np.array(xpositionA_traj[i])-np.array(xpositionB_traj[i]),np.linspace(1,num_steps,num_steps))
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
    x1 = data1_init
    y1 = yposition
    x2 = data2_init
    y2 = yposition
    
    # Initial line plots
    line1, = ax.plot(x1, y1, color='b', marker='.', markersize=1, label='DNA strand A')
    line2, = ax.plot(x2, y2, color='r', marker='.', markersize=1, label='DNA strand B')
    
    # Create a text label for the frame number, initially set to the first frame
    frame_text = ax.text(0.05, 0.95, f"Frame: {selected_frames[0]}", transform=ax.transAxes)
    
    ax.legend()
    
    def update(frame):
        # Fetch data for the current frame directly
        x1 = [xpositionA_traj[n][frame] for n in range(nsegs)]
        x2 = [xpositionB_traj[n][frame] for n in range(nsegs)]
        
        # Update line plots with new data
        line1.set_data(x1, y1)
        line2.set_data(x2, y2)
        
        # Update the frame number text
        frame_text.set_text(f"Frame: {selected_frames[frame]}")
        
        return line1, line2, frame_text

    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)
    
    # Save the animation as an MP4 file (uncomment to save)
    ani.save('./Data_outputs/1d_line_animation.gif', writer=PillowWriter(fps=20))
    
    # Show the plot
    plt.show()