import os 
import sys 
sys.path.append('../lznestpy')

import nestpy
from random import choices
from nestUtils import *

import matplotlib.pyplot as plt
import numpy as np


def simulate_LXe(EventRateFile, nEvents, g1, g2, interactionType=nestpy.INTERACTION_TYPE(0)):
    '''
    Event rate file: output from MMA, first column is energy, second column is differential event rate [keV-1 tonne-1 year-1].
    The delimiter is comma.

    nEvents: number of events to simulate. 

    g1: gain of S1, dimensionless. For LZ SR1, g1 = 0.114 . 
    g2: gain of S2, dimensionless. For LZ SR1, g2 = 47.1 .
    '''
    EventRate_file = np.loadtxt(EventRateFile, skiprows=0,delimiter=',')

    # Extract x and y values from the loaded data
    x_values = EventRate_file[:,0]  # recoil energy [keV]
    y_values = EventRate_file[:,1]  # event rate [keV-1 tonne-1 year-1]

    total_rate = np.trapz(y_values, x_values)

    weightPerEvent = total_rate / nEvents # Contribution of each event when making the final histogram

    sampled_energies = choices(x_values, weights=y_values, k=nEvents)

    ########################################
    ### Initialize LXE TPC detector
    ########################################
    detector = nestpy.LZ_Detector()
    detector.SetSR1Configuration()

    detector.set_s2_thr(0.)

    field = 192 # V/cm
    nc = nestpy.NESTcalc( detector )
    interactionType = nestpy.INTERACTION_TYPE(0) # 0 for NR, 8 for betas      

    ########################################
    ### Simulate events
    ########################################
    quantas = generate_quanta(NESTcalc= nc, 
                              interaction= interactionType, 
                              energy_array= sampled_energies, 
                              field= field, 
                              density = 2.9)

    random_R2 = np.random.uniform( 0., 698.*698., nEvents )
    random_phi = np.random.uniform( 0., 2.*np.pi, nEvents )
    random_X = np.sqrt( random_R2 ) * np.cos( random_phi )
    random_Y = np.sqrt( random_R2 ) * np.sin( random_phi )

    random_driftTimes = np.random.uniform( 0., 931., nEvents )

    driftVelocity = nc.SetDriftVelocity(detector.get_T_Kelvin(), 2.9, field )
    random_Z = detector.get_TopDrift() - random_driftTimes*driftVelocity

    S1s, spikes = generate_S1_highEnergy(nc, 
                            interactionType, 
                            sampled_energies, 
                            field, 
                            quantas, 
                            random_X, 
                            random_Y, 
                            random_Z, 
                            driftVelocity)

    S2s = (generate_S2(nc, 
                        interactionType, 
                        sampled_energies, 
                        field, 
                        quantas, 
                        random_X, 
                        random_Y, 
                        random_Z, 
                    driftVelocity, 
                    random_driftTimes ))


    reconstructed_energies = 1E-3*13.7*(S1s/g1 + S2s/g2)

    return reconstructed_energies, S1s, spikes, S2s, weightPerEvent*np.ones(nEvents)

