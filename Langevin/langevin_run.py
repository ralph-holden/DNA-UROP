# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:20:05 2024

@author: 44775
"""
# # # IMPORTS # # #
from langevin_model import Grain, Strand, Simulation, Start_position, np, Tuple, combinations
from langevin_model import kb, temp, kappab, lp, dt, gamma, homology_set
import pickle
import sys
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter



# # # SIMULATION PARMAMETERS # # #
# Run the Monte Carlo algorithm for given number of steps with a progress bar
nsteps = 1500
# Length of Segments, where each segment/grain is 1/5 helical coherence length
coherence_lengths = 10
curved = False
nsegs = 5 * coherence_lengths 
ystart = coherence_lengths/(5*np.pi) if curved else -1*coherence_lengths/2
# Separation, surface to surface (along x axis)
sep = 5
sep += 0.2 # augment for surface to surface
xstartA, xstartB = -sep/2, +sep/2
# Box Limits
xlim, ylim, zlim = 10, 10, 10 # from -lim to +lim 



# # # DATA OUTPUT PARAMETERS # # #
# save data?
save_data = False
log_update = 100 # how often to publish values to the log file
# animation?
animate = True
frame_hop = 50 # frame dump frequency



# # # INITIALISE & RUN SIMULATION # # #
spA = Start_position(nsegs, xstartA-10, ystart - 10, 0)
Strand1 = spA.create_strand_curved() if curved else spA.create_strand_straight()

spB = Start_position(nsegs, xstartB, ystart, 0)
Strand2 = spB.create_strand_curved() if curved else spB.create_strand_straight()

sim = Simulation(StrandA=Strand1, StrandB=Strand2, boxlims=np.array([xlim,ylim,zlim]))


for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
log_filename = datetime.now().strftime('./Data_outputs/LOG_%Y%m%d_%H%M%S.log')
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')
logging.info('Simulation started')
logging.info(f'''Simulation parameters:
    gamma   : {gamma}
    dt      : {dt}
    nsteps  : {nsteps}
    boxlims : {xlim}, {ylim}, {zlim}
    homology: {homology_set} 
Starting conditions:
    separation: {sep-0.2} (surface to surface)
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
        endtoendA, endtoendB = sim.endtoends[-1][0], sim.endtoends[-1][1]
        logging.info(f'''Step {i} : DATA:
Strand A end to end = {endtoendA} lc
Strand B end to end = {endtoendB} lc
Total Pairs = {sim.pair_counts[-1][0]}
Number of Islands = {sim.n_islands[-1]}
Simulation Internal Energy = {sim.energies[-1]}
Island Pair Distance = {sim.av_R_islands[-1]}
Average Island Length = {sim.av_L_islands[-1]}
Average Island Separation = {sim.av_sep_islands[-1]}
...''')
        print(f'''\rStrand A end to end = {endtoendA} lc
Strand B end to end = {endtoendB} lc
Total Pairs = {sim.pair_counts[-1][0]}
Simulation Total Energy = {sim.energies[-1]} 
...''')
    
# save trajectories
if save_data:
    with open('./Data_outputs/test_simulation.dat','wb') as data_f:
        pickle.dump([sim.trajectoryA, sim.trajectoryB], data_f)
        

# extracting data from trajectories
xsteps = np.linspace(0,len(sim.trajectoryA),len(sim.trajectoryA))
endtoendA, endtoendB = np.array(sim.endtoends)[:,0], np.array(sim.endtoends)[:,1]
endtoendA, endtoendB = list(endtoendA), list(endtoendB)
totpair, selfpair = np.array(sim.pair_counts)[:,0], np.array(sim.pair_counts)[:,1]
totpair, selfpair = list(totpair), list(selfpair)

# plotting end to end distances
endtoendendA, endtoendendB = sim.endtoend(-1)
print()
print(f'End to end distance Strand A = {endtoendendA}')
print(f'End to end distance Strand B = {endtoendendB}')

plt.figure()
plt.title('Coarse Grain DNA End to End Distance')
plt.xlabel('Timestep')
plt.ylabel('End to End distance, $l_c$')
plt.plot(xsteps, endtoendA, label = 'Strand A')
plt.plot(xsteps, endtoendB, label = 'Strand B')
plt.grid(linestyle=':')
plt.legend(loc='best')
plt.savefig('./Data_outputs/endtoend.png')
plt.show()

# plotting number of pairs
totpair_end, selfpair_end = sim.pair_counts[-1][0], sim.pair_counts[-1][1]
print()
print(f'Total number of paired grains = {totpair_end}')
print(f'Number of paired grains to self = {selfpair_end}')
print(f'Number of Islands = {sim.n_islands[-1]}')

plt.figure()
plt.title('Coarse Grain DNA Pairs')
plt.xlabel('Timestep')
plt.ylabel('Paired DNA Grains, $0.2 l_c$')
plt.plot(xsteps, totpair, label = 'Total')
plt.plot(xsteps, selfpair, label = 'Self')
plt.plot(xsteps, sim.n_islands, label = 'Islands')
plt.grid(linestyle=':')
plt.legend(loc='best')
plt.savefig('./Data_outputs/pairs.png')
plt.show()

# plotting total internal energy
print()
print(f'Internal Energy = {sim.energies[-1]} kbT')

plt.figure()
plt.title('Coarse Grain DNA Internal Energy')
plt.xlabel('Timestep')
plt.ylabel('Energy, $k_bT$')
plt.plot(xsteps, sim.energies, label='')
plt.grid(linestyle=':')
plt.legend(loc='best')
plt.savefig('./Data_outputs/energy.png')
plt.show()

# plotting average island pair distance
print()
print(f'Average Island Pair Distance = {sim.av_R_islands[-1]}')

plt.figure()
plt.title('Average Island Pair Distance')
plt.xlabel('Timestep')
plt.ylabel('R, $l_c$')
plt.plot(xsteps, sim.av_R_islands, label='')
plt.grid(linestyle=':')
plt.legend(loc='best')
plt.savefig('./Data_outputs/island_R.png')
plt.show()

# plotting average island length
print()
print(f'Average Island Length = {sim.av_L_islands[-1]}')

plt.figure()
plt.title('Average Island Length')
plt.xlabel('Timestep')
plt.ylabel('L, $l_c$')
plt.plot(xsteps, sim.av_L_islands, label='')
plt.grid(linestyle=':')
plt.legend(loc='best')
plt.savefig('./Data_outputs/island_length.png')
plt.show()

# plotting average island separation
print()
print(f'Average Island Separation = {sim.av_sep_islands[-1]}')

plt.figure()
plt.title('Average Island Separation')
plt.xlabel('Timestep')
plt.ylabel('L, $l_c$')
plt.plot(xsteps, sim.av_sep_islands, label='')
plt.grid(linestyle=':')
plt.legend(loc='best')
plt.savefig('./Data_outputs/island_separation.png')
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
    ani.save('./Data_outputs/3d_line_animation.gif', writer=PillowWriter(fps=20))
    
    logging.info('Animation saved as gif')
    
    # Show the plot
    plt.show()
    
logging.info('Simulation completed')
