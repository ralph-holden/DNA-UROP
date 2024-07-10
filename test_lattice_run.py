"""
Created on Sat Jun 22 10:53:49 2024

@author: 44775 Ralph Holden
"""
from lattice_dna_model import *

import pickle
import sys

test = lattice_dna(100, 100, 100, [-25, -50, 0], [25, -50, 0], 10)

# Run the Monte Carlo algorithm for given number of steps with a progress bar
nsteps = 50000

for i, item in enumerate(range(nsteps)):
    test.montecarlostep_gen2()
    
    length = 20
    progress = (i + 1) / nsteps
    bar_length = int(length * progress)
    bar = f"[{'=' * bar_length:{length}}] {progress * 100:.1f}%"
    sys.stdout.write(f"\r{bar}")
    sys.stdout.flush()
    
print()
print('dnastr_A')
print(test.dnastr_A)
print()
print('dnastr_B')
print(test.dnastr_B)

print()
print('for DNA A, the final end to end distance is', test.end_to_end()[0])
print('for DNA B, the final end to end distance is', test.end_to_end()[1])
print()
print('total number of paired sites', test.total_adj()[0])

test.proj_2d(fullbox = True )
test.proj_2d(fullbox = False)

test.proj_3d()

with open('test_simulation.dat','wb') as data_f:
    pickle.dump(test.trajectories, data_f)
    
    
# # # animation # # #
    
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Assuming 'test.trajectories' is defined and contains your data
t = test.trajectories

# Number of frames for the animation
num_frames = len(t)

# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Precomputed data (replace these lists with your actual data)
# Each element in the list should be a tuple (x, y, z)
data1 = [(np.array(t[i][0])[:, 0], np.array(t[i][0])[:, 1], np.array(t[i][0])[:, 2]) for i in range(num_frames)]
data2 = [(np.array(t[i][1])[:, 0], np.array(t[i][1])[:, 1], np.array(t[i][1])[:, 2]) for i in range(num_frames)]

# Initial data
x1, y1, z1 = data1[0]
x2, y2, z2 = data2[0]

# Initial line plots
line1, = ax.plot(x1, y1, z1, color='b', marker='.', label='Set 1')
line2, = ax.plot(x2, y2, z2, color='r', marker='.', label='Set 2')

# Set consistent axis limits
ax.set_xlim(-30, 30)
ax.set_ylim(-60, -20)
ax.set_zlim(-20, 20)
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
ani = FuncAnimation(fig, update, frames=num_frames, interval = 400,blit=False)

# Save the animation as an MP4 file (uncomment to save)
# ani.save('3d_line_animation.mp4', writer='ffmpeg', fps=10)

# Show the plot
plt.show()
