# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 10:53:49 2024

@author: 44775 Ralph Holden
"""
from free_dna_model import *

import numpy as np
import matplotlib.pyplot as plt

import pickle
import sys

s = Simulation(boxlims=Vector(50,50,50), StrandA=Strand(20,Vector(-5,0,0)), StrandB=Strand(20,Vector(5,0,0)))

# Run the Monte Carlo algorithm for given number of steps with a progress bar
nsteps = 1000

for i, item in enumerate(range(nsteps)):
    s.montecarlostep()
    
    length = 20
    progress = (i + 1) / nsteps
    bar_length = int(length * progress)
    bar = f"[{'=' * bar_length:{length}}] {progress * 100:.1f}%"
    sys.stdout.write(f"\r{bar}")
    sys.stdout.flush()
    
# convert to numpy arrays
s.trajectoryA = np.array(s.trajectoryA)
s.trajectoryB = np.array(s.trajectoryB)    

print()
print('Final DNA Strand A')
print(s.trajectoryA)
print()
print('Final DNA Strand B')
print(s.trajectoryB)


# # # animation # # #
    
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Assuming 's.trajectoryA' and 's.trajectoryB' are defined and contain your data
tA = s.trajectoryA
tB = s.trajectoryB

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
line1, = ax.plot(x1, y1, z1, color='b', marker='.', label='Set 1')
line2, = ax.plot(x2, y2, z2, color='r', marker='.', label='Set 2')

# Set consistent axis limits
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_zlim(-50, 50)
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
# ani.save('3d_line_animation.mp4', writer='ffmpeg', fps=10)

# Show the plot
plt.show()
