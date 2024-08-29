# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:00:46 2024

@author: ralph
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann

from Electrostatics_classholder import Electrostatics

elstats = Electrostatics()

ishomol = False

#R = np.linspace(0.19e-8 , 0.6e-8, 1000)
R = np.linspace(0.2e-8 , 0.6e-8, 1000)

a0, a1, a2 = elstats.a(R)

h = 0.0001e-8

a0_ph, a1_ph, a2_ph = elstats.a(R+h)
a0_mh, a1_mh, a2_mh = elstats.a(R-h)

SP_min_factor = []
SP_min_factor_ph = []
SP_min_factor_mh = []
for i in range(len(a1)):
    a11 = a1[i]
    a22 = a2[i]
    SP_min_factor += [np.cos(np.arccos(np.clip(a11/(4*a22), -1, 1)))] if abs(a22) > 10**-10 else [np.cos(np.pi)] 
    a11_ph = a1_ph[i]
    a22_ph = a2_ph[i]
    SP_min_factor_ph += [np.cos(np.arccos(np.clip(a11_ph/(4*a22_ph), -1, 1)))] if abs(a22_ph) > 10**-10 else [np.cos(np.pi)]
    a11_mh = a1_mh[i]
    a22_mh = a2_mh[i]
    SP_min_factor_mh += [np.cos(np.arccos(np.clip(a11_mh/(4*a22_mh), -1, 1)))] if abs(a22_mh) > 10**-10 else [np.cos(np.pi)] # avoid dividing by zero error
SP_min_factor = np.array(SP_min_factor)
SP_min_factor_ph = np.array(SP_min_factor_ph)
SP_min_factor_mh = np.array(SP_min_factor_mh)

fig1 = plt.figure()
ax1 = fig1.add_subplot()
fig2 = plt.figure()
ax2 = fig2.add_subplot()
fig3 = plt.figure()
ax3 = fig3.add_subplot()

for Lindex in [1,2,3,4,5]:
    Lmin = 20  * 10**-10
    Lmax = 100000 * 10**-10
    Lstep = 20 * 10**-10 # grain diameter
    Lrange = np.linspace( Lmin, Lmax, int((Lmax-Lmin)/(Lstep))+1 )
    L = Lrange[Lindex-1] if not ishomol else 1
    L0 = 18e-8
    
    if not ishomol: 
        Eint = elstats.coeffs * ( a0  -  elstats.nu(1, L) * a1 * SP_min_factor  +  elstats.nu(2, L) * a2 * SP_min_factor )
        Eint *= L0**Lindex
        
        # compare different methods manually
        if False:
            Eint_phpL = elstats.coeffs * ( a0_ph  -  elstats.nu(1, L+h) * a1_ph * SP_min_factor_ph  +  elstats.nu(2, L+h) * a2_ph * SP_min_factor_ph ) * (L+h)
            Eint_phmL = elstats.coeffs * ( a0_ph  -  elstats.nu(1, L-h) * a1_ph * SP_min_factor_ph  +  elstats.nu(2, L-h) * a2_ph * SP_min_factor_ph ) * (L-h)
            Eint_mhpL = elstats.coeffs * ( a0_mh  -  elstats.nu(1, L+h) * a1_mh * SP_min_factor_mh  +  elstats.nu(2, L+h) * a2_mh * SP_min_factor_mh ) * (L+h)
            Eint_mhmL = elstats.coeffs * ( a0_mh  -  elstats.nu(1, L-h) * a1_mh * SP_min_factor_mh  +  elstats.nu(2, L-h) * a2_mh * SP_min_factor_mh ) * (L-h)
            Force = -1 * (Eint_phpL + Eint_mhmL - Eint_mhpL - Eint_phmL)/ ( 4*h**2 )
        if False:
            Eint_ph = elstats.coeffs * ( a0_ph  -  elstats.nu(1, L) * a1_ph * SP_min_factor_ph  +  elstats.nu(2, L) * a2_ph * SP_min_factor_ph ) * (L)
            Force = -1 * (Eint_ph - Eint)/ (h)
        if True:
            Force = -np.gradient(Eint)
        
    elif ishomol:
        Eint = elstats.coeffs * ( a0  -  a1 * SP_min_factor  +  a2 * SP_min_factor ) 
        Eint *= L0**Lindex
    
        if True:
            Force = -np.gradient(Eint)
    
    Eint /= Boltzmann
    Eint /= 1e8 #** (Lindex-1)
    Force /= Boltzmann
    Force /= 1e8 #** (Lindex)
    a0 *= elstats.coeffs
    a1 *= elstats.coeffs
    a2 *= elstats.coeffs
    

    ax2.plot(R, Eint,label=f'L = {L}')
    ax3.plot(R, Force,label=f'L = {L}')
    
    
ax2.set_title('Eint vs R')
ax2.legend(loc='best')
mulitplier = 1000
#ax2.set_ylim(-2*mulitplier,+2*mulitplier)
ax3.set_title('Force vs R')
ax3.legend(loc='best')
#ax3.set_ylim(-50*mulitplier,+50*mulitplier)
    
ax1.plot(R, a0, label='a0')
ax1.plot(R, a1, label='a1')
ax1.plot(R, a2, label='a2')
ax1.set_title('a vs R')
ax1.legend(loc='best')
