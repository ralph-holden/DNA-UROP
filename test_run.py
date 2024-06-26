# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 10:53:49 2024

@author: 44775 Ralph Holden
"""
from dna_str import *

import pickle
import sys

test = dna_string_model(100, 100, 100, [-2, -50, 0], [2, -50, 0], 1000)

# Run the Monte Carlo algorithm for given number of steps with a progress bar
nsteps = 1000

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
