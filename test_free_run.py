# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 13:06:51 2024

@author: 44775 Ralph Holden
"""
# # # IMPORTS # # #
from free_dna_model import Vector, Bead, Strand, Simulation, np, Tuple, combinations
import pickle
import sys
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter


# # # SIMULATION PARMAMETERS # # #
# Run the Monte Carlo algorithm for given number of steps with a progress bar
nsteps = 1000
# Length of Segments
nsegs = 30
ystart = -nsegs/2
# Separation (along x axis)
sep = 10
xstartA, xstartB = -sep/2, +sep/2
# Box Limits
xlim, ylim, zlim = 20, 20, 20 # from -lim to +lim 

# save data?
save_data = False
log_update = 50 # how often to publish values to the log file
# animation?
animate = False

# # # INITIALISE & RUN SIMULATION # # #
sim = Simulation(boxlims=Vector(xlim,ylim,zlim), StrandA=Strand(nsegs,Vector(xstartA,ystart,0)), StrandB=Strand(nsegs,Vector(xstartB,ystart,0)))

for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
log_filename = datetime.now().strftime('./Data_outputs/LOG_%Y%m%d_%H%M%S.log')
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')
logging.info('Simulation started')

for i, item in enumerate(range(nsteps)):
    sim.montecarlostep_trial()
    
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
Simulation Total Energy = {sim.free_energy} kbT
...''')
        print(f'''\rStrand A end to end = {endtoendA} lc
Strand B end to end = {endtoendB} lc
Simulation Total Energy = {sim.free_energy} kbT
...''')
    
# save trajectories
if save_data:
    with open('./Data_outputs/test_simulation.dat','wb') as data_f:
        pickle.dump([sim.trajectoryA, sim.trajectoryB], data_f)
    
# convert to numpy arrays
sim.trajectoryA = np.array(sim.trajectoryA)
sim.trajectoryB = np.array(sim.trajectoryB)    

# extracting data from trajectories
xsteps = np.linspace(0,sim.nsteps,sim.nsteps)
endtoendA = []
endtoendB = []
for t in range(len(sim.trajectoryA)):
    a, b = sim.endtoend(t)
    endtoendA.append(a), endtoendB.append(b)

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
plt.show()

# plotting total free energy
print()
print(f'Free Energy = {sim.free_energy} kbT')

plt.figure()
plt.title('Coarse Grain DNA Free Energy')
plt.xlabel('Monte Carlo Step')
plt.ylabel('Free Energy, $k_bT$')
plt.plot(xsteps, sim.fe_traj, label='')
plt.grid(linestyle=':')
plt.legend(loc='best')
plt.show()
    
# # # ANIMATION # # #
if animate:
    # data
    tA = sim.trajectoryA
    tB = sim.trajectoryB
    
    # Number of frames for the animation
    num_frames = len(tA)
    
    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Precomputed data (replace these lists with your actual data)
    # Each element in the list should be a tuple (x, y, z)
    data1 = [(tA[i][:, 0], tA[i][:, 1], tA[i][:, 2]) for i in range(num_frames)]
    data2 = [(tB[i][:, 0], tB[i][:, 1], tB[i][:, 2]) for i in range(num_frames)]
    
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
    
logging.info('Simulation completed')
