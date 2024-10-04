# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:20:05 2024

@author: Ralph Holden
"""
# # # IMPORTS # # #
from langevin_model import Grain, Strand, Simulation, Start_position, np, Tuple, combinations
from langevin_model import kb, temp, kappab, lp, dt, gamma, homology_set, xi
import pickle
import sys
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter



# # # SIMULATION PARMAMETERS # # #
# Run the Monte Carlo algorithm for given number of steps with a progress bar
nsteps = 10000
# Length of Segments, where each segment/grain is 1/5 helical coherence length
coherence_lengths = 5
curved = False
nsegs = 5 * coherence_lengths 
ystart = coherence_lengths/(2*np.pi) if curved else -1*coherence_lengths*10**-8/2
# Separation, surface to surface (along x axis)
sep = 0.22e-8
#sep += 0.2 # augment for surface to surface
xstartA, xstartB = -sep/2, +sep/2
# starting shift 
yshift = 0.0


# # # DATA OUTPUT PARAMETERS # # #
# data output directory
mydir = './Data_outputs/test_update/'
if not os.path.exists(mydir):
    os.makedirs(mydir)
# save data
save_data = True
log_update = 100 # how often to publish values to the log file

# terminating settings
recall_steps = 5000
ignore_steps = 2000 + recall_steps
std_tol = 0.01 

# animation
animate = True
frame_hop = 20 # frame dump frequency
xlim, ylim, zlim = 2e-8, 12e-8, 2e-8 # Box Limits, for viewing, from -lim to +lim



# # # INITIALISE & RUN SIMULATION # # #
spA = Start_position(nsegs, xstartA, ystart+yshift, 0)
Strand1 = spA.create_strand_curved() if curved else spA.create_strand_straight()

# rewrite settings to have one curved strand
#curved = True
#ystart = coherence_lengths/(2*np.pi) if curved else -1*coherence_lengths/2

spB = Start_position(nsegs, xstartB, ystart, 0)
Strand2 = spB.create_strand_curved() if curved else spB.create_strand_straight()

sim = Simulation(StrandA=Strand1, StrandB=Strand2, boxlims=np.array([xlim,ylim,zlim]))


for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
log_filename = datetime.now().strftime(mydir+'LOG_%Y%m%d_%H%M%S.log')
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')
logging.info('Simulation started')
logging.info(f'''Simulation parameters:
    gamma    : {gamma}
    xi       : {xi}
    dt       : {dt}
    nsteps   : {nsteps}
    num parts: {nsegs}
    num l_c  : {coherence_lengths}
    boxlims  : {xlim}, {ylim}, {zlim}
    homology : {homology_set} 
Starting conditions:
    separation: {sep} (interaxial)
    curvature : {curved}
             ''')

for i, item in enumerate(range(nsteps)):
    sim.run_step()
    
    length = 20
    progress = (i + 1) / nsteps
    bar_length = int(length * progress)
    bar = f"[{'=' * bar_length:{length}}] {progress * 100:.1f}%"
    sys.stdout.write(f"\r{bar}")
    sys.stdout.flush()
    
    # log file & data output
    if i % log_update == 0:
        endtoendA, endtoendB = sim.endtoend_traj[-1][0], sim.endtoend_traj[-1][1]
        logging.info(f'''Step {i} : DATA:
Simulation Internal Energy = {sim.energy_traj[-1]} kbT

Strand A end to end = {endtoendA} m
Strand B end to end = {endtoendB} m
Mean Curvature      = {sim.mean_curvature_traj[-1]*180/np.pi} degrees
STD  Curvature      = {sim.std_curvature_traj[-1]*180/np.pi} degrees
Number Loops        = {sim.n_loops_traj[-1]}

Total Pairs              = {sim.total_pairs_traj[-1]}
Homologous Pairs         = {sim.homol_pairs_traj[-1]}
Homologous Pair Distance = {sim.homol_pair_dist_traj[-1]} m
Number Islands           = {sim.n_islands_traj[-1]}

...''')
        print(f'''\rSimulation Internal Energy = {sim.energy_traj[-1]}
Strand A end to end        = {endtoendA} lc
Strand B end to end        = {endtoendB} lc
Mean Curvature             = {sim.mean_curvature_traj[-1]*180/np.pi} degrees
STD  Curvature             = {sim.std_curvature_traj[-1]*180/np.pi} degrees
Number Loops               = {sim.n_loops_traj[-1]}
Total Pairs                = {sim.total_pairs_traj[-1]}
Homologous Pairs           = {sim.homol_pairs_traj[-1]}
Homologous Pair Distance   = {sim.homol_pair_dist_traj[-1]} m
Number Islands             = {sim.n_islands_traj[-1]}
...''')
        if not endtoendA < 1 and not endtoendA > 1 or not endtoendB < 1 and not endtoendB > 1: #always True if 'nan'
            error_msg = f'STEP {i}: Simulation terminating - lost grains'
            print(error_msg)
            logging.info(error_msg)
            break # end simulation
            
        if i>ignore_steps and np.std( sim.homol_pairs_traj[-recall_steps:] )/sim.StrandA.num_segments < std_tol:
            finish_msg = f'STEP {i}: Simulation terminating - pairs converged'
            print(finish_msg)
            logging.info(finish_msg)
            break # end simulation
        
# extracting data from trajectories
xsteps = np.linspace(0,len(sim.trajectoryA),len(sim.trajectoryA))
endtoendA, endtoendB = np.array(sim.endtoend_traj)[:,0], np.array(sim.endtoend_traj)[:,1]
endtoendA, endtoendB = list(endtoendA), list(endtoendB)

# plotting end to end distances
endtoendendA, endtoendendB = sim.find_endtoend(-1)
print()
print(f'End to end distance Strand A = {endtoendendA}')
print(f'End to end distance Strand B = {endtoendendB}')

plt.figure()
plt.title('Coarse Grain DNA End to End Distance')
plt.xlabel(f'Timestep, {dt}')
plt.ylabel('End to End distance, $m$')
plt.plot(xsteps, endtoendA, label = 'Strand A')
plt.plot(xsteps, endtoendB, label = 'Strand B')
plt.grid(linestyle=':')
plt.legend(loc='best')
plt.savefig(mydir+'endtoend.png')
plt.show()

# plotting total internal energy
print()
print(f'Internal Energy = {sim.energy_traj[-1]} kbT')

plt.figure()
plt.title('Coarse Grain DNA Internal Energy')
plt.xlabel(f'Timestep, {dt}')
plt.ylabel('Energy, $k_bT$')
plt.plot(xsteps, sim.energy_traj)
plt.grid(linestyle=':')
plt.savefig(mydir+'energy.png')
plt.show()

# plotting curvature
print()
print(f'Mean Curvature = {sim.mean_curvature_traj[-1]*180/np.pi} degrees')
print(f'STD  Curvature = {sim.std_curvature_traj[-1]*180/np.pi} degrees')

plt.figure()
plt.title('Mean Curvature')
plt.xlabel(f'Timestep, {dt}')
plt.ylabel('Curvature, degrees')
plt.plot(xsteps, abs(np.array(sim.mean_curvature_traj)*180/np.pi) + np.array(sim.std_curvature_traj)*180/np.pi, label='+std', color='orange')
plt.plot(xsteps, abs(np.array(sim.mean_curvature_traj)*180/np.pi), label='mean')
plt.plot(xsteps, abs(np.array(sim.mean_curvature_traj)*180/np.pi) - np.array(sim.std_curvature_traj)*180/np.pi, label='-std', color='orange')
plt.grid(linestyle=':')
plt.legend(loc='best')
plt.savefig(mydir+'curvature.png')
plt.show()

# plotting pair data
print()
print(f'Total Pairs = {sim.total_pairs_traj[-1]}') # may want to remove
print(f'Homologous Pairs = {sim.homol_pairs_traj[-1]}')
print(f'Homologous Pair Distance = {sim.homol_pair_dist_traj[-1]} m')
print(f'Number islands = {sim.n_islands_traj[-1]}')

plt.figure(figsize=[16,5])

plt.subplot(1, 2, 1)
plt.title('Pair Number')
plt.xlabel(f'Timestep, {dt}')
plt.ylabel('Number of Pairs')
plt.plot(xsteps, sim.total_pairs_traj, label='Total Pairs')
plt.plot(xsteps, sim.homol_pairs_traj, label='Homologous Pairs')
plt.plot(xsteps, sim.n_loops_traj,     label='Loops')
plt.grid(linestyle=':')
plt.legend(loc='best')

plt.subplot(1, 2, 2)
plt.title('Homologous Pair Separation')
plt.xlabel(f'Timestep, {dt}')
plt.ylabel('Distance, $m$')
plt.plot(xsteps, sim.homol_pair_dist_traj, label='All homologous pairs')
plt.plot(xsteps, sim.terminal_dist_traj, label='End homologous pairs')
plt.grid(linestyle=':')
plt.legend(loc='best')

plt.savefig(mydir+'pair_data.png')
plt.show()

# plotting average island length & separation
print()
print(f'Average Island Length = {sim.L_islands_traj[-1]}')
print(f'Average Island Separation = {sim.sep_islands_traj[-1]}')

plt.figure(figsize=[16,10])

plt.subplot(2, 2, 1)
plt.title('Average Island Length')
plt.xlabel(f'Timestep, {dt}')
plt.ylabel('L, $m$')
plt.plot(xsteps, sim.L_islands_traj)
plt.grid(linestyle=':')

plt.subplot(2, 2, 2)
plt.title('Average Island Separation')
plt.xlabel(f'Timestep, {dt}')
plt.ylabel('L, $m$')
plt.plot(xsteps, sim.sep_islands_traj)
plt.grid(linestyle=':')

plt.subplot(2, 2, 3)
plt.title('Average Island Number')
plt.xlabel(f'Timestep, {dt}')
plt.ylabel('Number of Islands')
plt.plot(xsteps, sim.n_islands_traj)
plt.grid(linestyle=':')

plt.subplot(2, 2, 4)
plt.title('Average Island Distance')
plt.xlabel(f'Timestep, {dt}')
plt.ylabel('Interaxial Separation, $m$')
plt.plot(xsteps, sim.R_islands_traj)
plt.grid(linestyle=':')

plt.savefig(mydir+'island_data.png')
plt.show()

    
# # # ANIMATION # # #
if animate:
    # data
    tA = np.array(sim.trajectoryA)
    tB = np.array(sim.trajectoryB)
    
    # Number of frames for the animation
    selected_frames = range(0,len(tA),frame_hop)
    num_frames = len(selected_frames)
    
    # Create a figure and a 3D axis
    fig = plt.figure(figsize=[16,10])
    ax = fig.add_subplot(111, projection='3d')
    
    # Precomputed data (replace these lists with your actual data)
    # Each element in the list should be a tuple (x, y, z)
    data1 = [(tA[i][:, 0], tA[i][:, 1], tA[i][:, 2]) for i in selected_frames]
    data2 = [(tB[i][:, 0], tB[i][:, 1], tB[i][:, 2]) for i in selected_frames]
    
    # Initial data
    x1, y1, z1 = data1[0]
    x2, y2, z2 = data2[0]
    
    # Initial line plots
    line1, = ax.plot(x1, y1, z1, color='b', marker='.', markersize=1, label='DNA strand A')
    line2, = ax.plot(x2, y2, z2, color='r', marker='.', markersize=1, label='DNA strand B')
    
    # Create a text label for the frame number, initially set to the first frame
    frame_text = ax.text2D(0.05, 0.95, f"Frame: {selected_frames[0]}", transform=ax.transAxes)
    
    # Axis limits
    ax.set_xlim(-sim.boxlims[0], sim.boxlims[0])
    ax.set_ylim(-sim.boxlims[1], sim.boxlims[1])
    ax.set_zlim(-sim.boxlims[2], sim.boxlims[2])
    ax.legend()
    
    # Update function for the animation
    def update(frame):
        # Fetch data for the current frame
        x1, y1, z1 = data1[frame]
        x2, y2, z2 = data2[frame]
        
        # Update line plots with new data
        line1.set_data_3d(x1, y1, z1)
        line2.set_data_3d(x2, y2, z2)
        
        # Update the frame number text
        frame_text.set_text(f"Frame: {selected_frames[frame]}")

        return line1, line2, frame_text
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)
    
    # Save the animation as an MP4 file (uncomment to save)
    ani.save(mydir+'3d_line_animation.gif', writer=PillowWriter(fps=20))
    
    logging.info('Animation saved as gif')
    
    # Show the plot
    plt.show()
    
# save trajectories
if save_data:
    with open(mydir+'test_simulation.dat','wb') as data_f:
        pickle.dump([sim.trajectoryA, sim.trajectoryB], data_f)

logging.info('Simulation completed')
