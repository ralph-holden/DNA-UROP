# -*- coding: utf-8 -*-
"""
Run a joint free Monte Carlo then langevin simulation

Created on Tue Jul 23 11:55:42 2024

@author: Ralph Holden
"""
# # # IMPORTS # # #
# python functions
import numpy as np
import pickle
import sys
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

# models
# add Free folder to the system path
sys.path.insert(0, '../FreeMC/')
# add Langevin folder to the system path
sys.path.insert(0, '../Langevin/')
import free_model as fm
import langevin_model as lm
from convert import convert



# # # F R E E  M O N T E  C A R L O # # #



# # # FREE MC SIMULATION PARMAMETERS # # #
# Run the Monte Carlo algorithm for given number of steps with a progress bar
nsteps = 100
# Length of Segments, where each segment/grain is 1/5 helical coherence length
nsegs = 5 * 30
ystart = -nsegs/5/2
# Separation (along x axis)
sep = 2
xstartA, xstartB = -sep/2, +sep/2
# Box Limits
xlim, ylim, zlim = 10, 20, 10 # from -lim to +lim 

# # # FREE MC DATA OUTPUT PARAMETERS # # #
# save data?
save_data = True
log_update = 10 # how often to publish values to the log file
# animation?
animate = True
frame_hop = 50 # frame dump frequency

# # # INITIALISE & RUN SIMULATION # # #
sim = fm.Simulation(boxlims=fm.Vector(xlim,ylim,zlim), StrandA=fm.Strand(nsegs,fm.Vector(xstartA,ystart,0)), StrandB=fm.Strand(nsegs,fm.Vector(xstartB,ystart,0)))

for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
log_filename = datetime.now().strftime('./Data_outputs/LOG_%Y%m%d_%H%M%S.log')
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')
logging.info('Simulation started')

for i, item in enumerate(range(nsteps)):
    sim.montecarlostep()
    
    length = 20
    progress = (i + 1) / nsteps
    bar_length = int(length * progress)
    bar = f"[{'=' * bar_length:{length}}] {progress * 100:.1f}%"
    sys.stdout.write(f"\r{bar}")
    sys.stdout.flush()
    
    # log file
    if i % log_update == 0:
        endtoendA, endtoendB = sim.endtoend(i)
        logging.info(f'''Step {i} : DATA:
Strand A end to end = {endtoendA} lc
Strand B end to end = {endtoendB} lc
Simulation Total Energy = {sim.energy} kbT
...''')
        print(f'''\rStrand A end to end = {endtoendA} lc
Strand B end to end = {endtoendB} lc
Simulation Total Energy = {sim.energy} kbT
...''')
    
# save trajectories
if save_data:
    with open('./Data_outputs/test_simulation.dat','wb') as data_f:
        pickle.dump([sim.trajectoryA, sim.trajectoryB], data_f)
    
# convert to numpy arrays
sim.trajectoryA = np.array(sim.trajectoryA)
sim.trajectoryB = np.array(sim.trajectoryB)    

# extracting data from trajectories
xsteps = np.linspace(0,sim.nsteps+1,sim.nsteps+1)
endtoendA = []
endtoendB = []
for t in range(len(sim.trajectoryA)):
    a, b = sim.endtoend(t)
    endtoendA.append(a), endtoendB.append(b)
totpair = np.array(sim.pair_count)[:,0]
selfpair = np.array(sim.pair_count)[:,1]

# plotting end to end distances
endtoendendA, endtoendendB = sim.endtoend(-1)
print()
print(f'End to end distance Strand A = {endtoendendA}')
print(f'End to end distance Strand B = {endtoendendB}')

plt.figure()
plt.title('Coarse Grain DNA End to End Distance')
plt.xlabel('Monte Carlo Step')
plt.ylabel('End to End distance, $l_c$')
plt.plot(xsteps, endtoendA, label = 'Strand A')
plt.plot(xsteps, endtoendB, label = 'Strand B')
plt.grid(linestyle=':')
plt.legend(loc='best')
plt.savefig('./Data_outputs/endtoend.png')
plt.show()

# plotting number of pairs
totpair_end, selfpair_end = sim.count_tot()
print()
print(f'Total number of paired grains = {totpair_end}')
print(f'Number of paired grains to self = {selfpair_end}')

plt.figure()
plt.title('Coarse Grain DNA Pairs')
plt.xlabel('Monte Carlo Step')
plt.ylabel('Paired DNA Grains, $0.2 l_c$')
plt.plot(xsteps, totpair, label = 'Total')
plt.plot(xsteps, selfpair, label = 'Self')
plt.grid(linestyle=':')
plt.legend(loc='best')
plt.savefig('./Data_outputs/pairs.png')
plt.show()

# plotting total free energy
print()
print(f'Free Energy = {sim.energy} kbT')

plt.figure()
plt.title('Coarse Grain DNA Free Energy')
plt.xlabel('Monte Carlo Step')
plt.ylabel('Free Energy, $k_bT$')
plt.plot(xsteps, sim.eng_traj, label='')
plt.grid(linestyle=':')
plt.legend(loc='best')
plt.savefig('./Data_outputs/energy.png')
plt.show()
    
# # # ANIMATION # # #
if animate:
    # data
    tA = sim.trajectoryA
    tB = sim.trajectoryB
    
    # Number of frames for the animation
    selected_frames = range(0,len(tA),frame_hop)
    num_frames = len(selected_frames)
    
    # Create a figure and a 3D axis
    fig = plt.figure()
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
    
    # Axis limits
    ax.set_xlim(-sim.boxlims.x, sim.boxlims.x)
    ax.set_ylim(-sim.boxlims.y, sim.boxlims.y)
    ax.set_zlim(-sim.boxlims.z, sim.boxlims.z)
    ax.legend()
    
    # Update function for the animation
    def update(frame):
        # Fetch data for the current frame
        x1, y1, z1 = data1[frame]
        x2, y2, z2 = data2[frame]
        
        # Update line plots with new data
        line1.set_data_3d(x1, y1, z1)
        line2.set_data_3d(x2, y2, z2)
        
        return line1, line2
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=num_frames, interval=400, blit=False)
    
    # Save the animation as an MP4 file (uncomment to save)
    ani.save('./Data_outputs/3d_line_animation.gif', writer=PillowWriter(fps=1))
    
    logging.info('Animation saved as GIF')
    
    # Show the plot
    plt.show()
    
logging.info('Free MC simulation completed')



# # # L A N G E V I N # # #


# # # LANGEVIN SIMULATION PARMAMETERS # # #
# Run the Monte Carlo algorithm for given number of steps with a progress bar
nsteps = 200

# # # LANGEVIN DATA OUTPUT PARAMETERS # # #
# save data?
save_data = False
log_update = 50 # how often to publish values to the log file
# animation?
animate = True
frame_hop = 1 # frame dump frequency, per logged trajectories


# # # INITIALISE & RUN SIMULATION # # #
#sim = lm.Simulation(lm.Strand(lm.gen_grains(coherence_lengths,[xstartA,ystart,0])), lm.Strand(lm.gen_grains(coherence_lengths,[xstartB,ystart,0])), boxlims=np.array([xlim,ylim,zlim]))
sim = convert(sim)

logging.info('Langevin simulation started')

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
        sim.record()
        endtoendA, endtoendB = sim.endtoends[-1][0], sim.endtoends[-1][1]
        logging.info(f'''Step {i} : DATA:
Strand A end to end = {endtoendA} lc
Strand B end to end = {endtoendB} lc
Simulation Total Energy = {sim.energies[-1]} 
...''')
        print(f'''\rStrand A end to end = {endtoendA} lc
Strand B end to end = {endtoendB} lc
Simulation Total Energy = {sim.energies[-1]} 
...''')
    
# save trajectories
if save_data:
    with open('./Data_outputs/test_simulation.dat','wb') as data_f:
        pickle.dump([sim.trajectoryA, sim.trajectoryB], data_f)
        

# extracting data from trajectories
xsteps = np.linspace(0,len(sim.trajectoryA)*log_update,len(sim.trajectoryA))
endtoendA, endtoendB = np.array(sim.endtoends)[:,0], np.array(sim.endtoends)[:,1]
totpair, selfpair = np.array(sim.pair_counts)[:,0], np.array(sim.pair_counts)[:,1]

# plotting end to end distances
endtoendendA, endtoendendB = sim.endtoend(-1)
print()
print(f'End to end distance Strand A = {endtoendendA}')
print(f'End to end distance Strand B = {endtoendendB}')

plt.figure()
plt.title('Coarse Grain DNA End to End Distance')
plt.xlabel('Monte Carlo Step')
plt.ylabel('End to End distance, $l_c$')
plt.plot(xsteps, endtoendA, label = 'Strand A')
plt.plot(xsteps, endtoendB, label = 'Strand B')
plt.grid(linestyle=':')
plt.legend(loc='best')
plt.savefig('./Data_outputs/endtoend.png')
plt.show()

# plotting number of pairs
totpair_end, selfpair_end = sim.count_tot()
print()
print(f'Total number of paired grains = {totpair_end}')
print(f'Number of paired grains to self = {selfpair_end}')

plt.figure()
plt.title('Coarse Grain DNA Pairs')
plt.xlabel('Monte Carlo Step')
plt.ylabel('Paired DNA Grains, $0.2 l_c$')
plt.plot(xsteps, totpair, label = 'Total')
plt.plot(xsteps, selfpair, label = 'Self')
plt.grid(linestyle=':')
plt.legend(loc='best')
plt.savefig('./Data_outputs/pairs.png')
plt.show()

# plotting total free energy
print()
print(f'Free Energy = {sim.energies[-1]} kbT')

plt.figure()
plt.title('Coarse Grain DNA Free Energy')
plt.xlabel('Monte Carlo Step')
plt.ylabel('Free Energy, $k_bT$')
plt.plot(xsteps, sim.energies, label='')
plt.grid(linestyle=':')
plt.legend(loc='best')
plt.savefig('./Data_outputs/energy.png')
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
    fig = plt.figure()
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
        line1.set_data(x1, y1)
        line1.set_3d_properties(z1)
        
        line2.set_data(x2, y2)
        line2.set_3d_properties(z2)
        
        return line1, line2

    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False)
    
    # Save the animation as an MP4 file (uncomment to save)
    ani.save('./Data_outputs/3d_line_animation.gif', writer=PillowWriter(fps=5))
    
    logging.info('Animation saved as GIF')
    
    # Show the plot
    plt.show()

logging.info('Langevin simulation completed')    
logging.info('Simulation completed')