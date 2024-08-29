# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 12:14:57 2024

@author: Ralph Holden
"""

# # # HOMOLOGY RECOGNITION FUNNEL # # #
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann
from free_model import Electrostatics

elstats = Electrostatics(homol=True)

def plot_energy(R,L):
    '''Plot homology recognition function as a function of displacement distance for strands L at length R'''
    
    a0, a1, a2 = elstats.a(R)
    
    lc = 1  *10**-8
    
    x = np.linspace(-5 *10**-8, 5 *10**-8,1000)
    
    zero_term = a0*( L - abs(x) )
    one_term = a1*( 1 - np.exp(-abs(x)/lc) + (L - 3*abs(x))/(2*lc) * np.exp(-abs(x)/lc) )
    two_term = a2*( 0.25*(1 - np.exp(-4*abs(x)/lc)) + (L - 3*abs(x))/(2*lc) * np.exp(-4*abs(x)/lc) )
    
    y = zero_term - 2*lc * ( one_term - two_term )
    y /= Boltzmann*300
    y *= 10**8
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title('Homology Recognition Funnel')
    ax.set_ylabel('Interaction Energy, $k_bT$')
    ax.set_xlabel('Displacement, x')
    text_kwargs = dict(ha='center', va='center', fontsize=10, color='C1',transform=ax.transAxes)
    ax.text(0.25, 0.8, f'R: {R}', **text_kwargs)
    ax.text(0.25, 0.7, f'L: {L}', **text_kwargs)
    plt.grid(linestyle=':')
    ax.plot(x,y)
    plt.show()
    
def plot_energy_3D(Xminmax, Rminmax, L, finite=False):
    
    lc = 1  *10**-8
    
    xrange = np.linspace(Xminmax[0],Xminmax[1])
    Rrange = np.linspace(Rminmax[0],Rminmax[1])
    
    xmesh, Rmesh = np.meshgrid(xrange, Rrange)
    
    a0, a1, a2 = elstats.a(Rmesh)
    
    if finite:
        zero_term = a0*( L - abs(xmesh) )
        one_term = a1*( 1 - np.exp(-abs(xmesh)/lc) + (L - 3*abs(xmesh))/(2*lc) * np.exp(-abs(xmesh)/lc) )
        two_term = a2*( 0.25*(1 - np.exp(-4*abs(xmesh)/lc)) + (L - 3*abs(xmesh))/(2*lc) * np.exp(-4*abs(xmesh)/lc) )
    
    if not finite:
        zero_term = a0*L
        one_term = a1*( 1 - np.exp(-abs(xmesh)/lc) + (L - 2*abs(xmesh))/(2*lc) * np.exp(-abs(xmesh)/lc) )
        two_term = a2*( 0.25*(1 - np.exp(-4*abs(xmesh)/lc)) + (L - 2*abs(xmesh))/(2*lc) * np.exp(-4*abs(xmesh)/lc) )
    
    E = zero_term - 2*lc * ( one_term - two_term )
    E /= Boltzmann*300
    E *= 10**8
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(xmesh, Rmesh, E, cmap='viridis', alpha=0.7)
    
    fig.colorbar(surf)
    
    ax.set_title('Homology Recognition Funnel')
    ax.set_zlabel('Interaction Energy, $k_bT$')
    ax.set_ylabel('R')
    ax.set_xlabel('Displacement, x')
    text_kwargs = dict(ha='center', va='center', fontsize=10, color='black',transform=ax.transAxes)
    ax.text2D(0.25, 0.8, f'L: {L}', **text_kwargs)
    plt.show()
    
plot_energy(R = 0.6 *10**-8 , L = 100 *10**-8)

plot_energy_3D( Xminmax = [-5 *10**-8, 5 *10**-8] , Rminmax = [0.6 *10**-8, 2 *10**-8] , L = 100 *10**-8, finite=False)

def delE(R, L):
    ''' Plot change in energy for full rotation as a function of L, for a value R '''
    
    a0, a1, a2 = elstats.a(R)
    
    lc = 1  *10**-8
    
    one_term = -a1*( 1 - np.exp(-L/(2*lc)) + L/(2*lc) )
    two_term = 0.25*a2*( (1 - np.exp(-2*L/lc)) + 2*L/lc )
    
    y = 2*lc * ( one_term + two_term )
    y /= Boltzmann*300
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title('Homology Recognition Funnel')
    ax.set_ylabel('$\Delta E(L)$, $k_bT$')
    ax.set_xlabel('L, m')
    text_kwargs = dict(ha='center', va='center', fontsize=10, color='C1',transform=ax.transAxes)
    ax.text(0.25, 0.8, f'R: {R}', **text_kwargs)
    plt.grid(linestyle=':')
    ax.plot(L,y)
    plt.show()
    
delE(R = 0.23 *10**-8 , L = np.linspace(1,100) ) 