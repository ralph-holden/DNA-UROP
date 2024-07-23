# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:08:10 2024

@author: Ralph Holden
"""
#from /Free/free_model import Vector, Bead, Strand, Simulation
import free_model as fm
import langevin_model as lm
import numpy as np

def convert(sim,fmtolm = True):
    '''
    Converts a simulation between Free MC and Langevin
    INPUTS:
        sim: class, Simulation of free_model or langevin_model
        fmtolm: bool, True for free to langevin, else does langevin to free
    OUPUTS:
        newsim: class, Simulation of the model specified
    '''
    
    if fmtolm:
        # box size
        xlim, ylim, zlim = sim.boxlims.x, sim.boxlims.y, sim.boxlims.z
        # read old (free MC model) beads, and use positions for new (langevin model) grains
        grainsA, grainsB = [], []
        for i in range(sim.StrandA.num_segments):
            grainsA.append(lm.Grain(sim.StrandA.dnastr[i].position.arr, np.zeros(3)))
            grainsB.append(lm.Grain(sim.StrandB.dnastr[i].position.arr, np.zeros(3)))
        # initialise simulation
        newsim = lm.Simulation(lm.Strand(grainsA), lm.Strand(grainsB), np.array([xlim,ylim,zlim]))
    
    else:
        # box size
        xlim, ylim, zlim = sim.boxlims[0], sim.boxlims[1], sim.boxlims[2]
        # initialise simulation
        newsim = fm.Simulation(fm.Vector(xlim,ylim,zlim), fm.Vector(-1,-ylim+1,0), fm.Vector(-1,-ylim+1,0))
        # read old (langevin model) grains, and use positions to overwrite new (free MC model) beads
        for i in range(sim.StrandA.num_segments):
            newsim.StrandA.dnastr[i].position = Vector(sim.StrandA.dnastr[i].position[0], sim.StrandA.dnastr[i].position[1], sim.StrandA.dnastr[i].position[2])
            newsim.StrandB.dnastr[i].position = Vector(sim.StrandB.dnastr[i].position[0], sim.StrandB.dnastr[i].position[1], sim.StrandB.dnastr[i].position[2])
    
    return newsim
       
    
        
    
