# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 10:53:49 2024

@author: 44775 Ralph Holden
"""
# # # IMPORTS # # #
from free_dna_model import Vector, Bead, Strand, Simulation, np, Tuple, combinations
import pickle
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter


# # # SIMULATION PARMAMETERS # # #
# Run the Monte Carlo algorithm for given number of steps with a progress bar
nsteps = 2000
# Length of Segments
nsegs = 30
ystart = -nsegs/2
# Separation (along x axis)
sep = 10
xstartA, xstartB = -sep/2, +sep/2
# Box Limits
xlim, ylim, zlim = 20, 20, 20 # from -lim to +lim 

# # # INITIALISE & RUN SIMULATION # # #
sim = Simulation(boxlims=Vector(xlim,ylim,zlim), StrandA=Strand(nsegs,Vector(xstartA,ystart,0)), StrandB=Strand(nsegs,Vector(xstartB,ystart,0)))

for i, item in enumerate(range(nsteps)):
    sim.montecarlostep_trial()
    
    length = 20
    progress = (i + 1) / nsteps
    bar_length = int(length * progress)
    bar = f"[{'=' * bar_length:{length}}] {progress * 100:.1f}%"
    sys.stdout.write(f"\r{bar}")
    sys.stdout.flush()
    
# save trajectories
with open('test_simulation.dat','wb') as data_f:
    pickle.dump([sim.trajectoryA, sim.trajectoryB], data_f)
    
# convert to numpy arrays
sim.trajectoryA = np.array(sim.trajectoryA)
sim.trajectoryB = np.array(sim.trajectoryB)    

print()
print('Final DNA Strand A')
print(sim.trajectoryA[-1])
print()
print('Final DNA Strand B')
print(sim.trajectoryB[-1])
    
# # # ANIMATION # # #
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
ani.save('3d_line_animation.gif', writer=PillowWriter(fps=5))

# Show the plot
plt.show()
