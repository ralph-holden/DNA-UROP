# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 10:53:49 2024

@author: 44775 Ralph Holden
"""

from dna_str import *

test = dna_string_model(100, 100, 100, [-50, -50, 0], [50, -50, 0], 100)

for i in range(100000):
    test.montecarlostep()
   
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
print('total number of paired sites', test.total_adj(), '<- clearly incorrect')


test.proj_2d()
