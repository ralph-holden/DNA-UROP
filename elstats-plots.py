# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 13:38:38 2024

@author: ralph
"""
import numpy as np
import matplotlib.pyplot as plt
from free_model import Electrostatics

lamb = 0.25
dt = 0.0001
temp = 310.15
kb = 1.38e-23

do3dplot = False

def do_the_plots():
    numfig = 4
    plt.figure()
    for i in range(1,numfig+1):
        R = i * 10**-10
        x = []
        y = []
        for l in range(1,50):
            x += [l]
            y += [elstats.force(l, R) * kb ]
        plt.subplot(2, 2, i)
        plt.title(f'Force from L, R={R}, homol: {homol_set}')
        plt.plot(x, y)
        plt.ylim(np.mean(y)-10**-12,np.mean(y)+10**-12)
    plt.tight_layout(pad=1)
    plt.show()
    
    numfig = 6
    plt.figure()
    for i in range(1,numfig+1):
        L = i
        x = []
        y = []
        for r in np.linspace(0.1*10**-8, 0.75*10**-8):
            x += [r]
            y += [elstats.force(L, r) * kb ]
        plt.subplot(3, 2, i)
        plt.title(f'Force from R, L={L}, homol: {homol_set}')
        plt.plot(x, y)
    plt.tight_layout(pad=1)
    plt.show()
    
homol_set = True
elstats = Electrostatics(homol=True)
do_the_plots()
if do3dplot:
    elstats.gen_energy_map()
    elstats.plot_energy_map()

homol_set = False
elstats = Electrostatics()
if do3dplot:
    elstats.gen_energy_map()
    elstats.plot_energy_map()
do_the_plots()


print('Velocity thermal fluctuation of langevin:')
print(np.sqrt(2 * lamb * kb * temp / dt))