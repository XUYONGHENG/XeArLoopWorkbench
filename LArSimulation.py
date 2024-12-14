import numpy as np
import matplotlib.pyplot as plt
from random import choices
import LArDS50 as LAr

SingleElectronS2 = 23

def simulate_LAr(EventRateFile, nEvents=1000000):
    '''
    Event rate file: output from MMA, first column is energy, second column is differential event rate [keV-1 tonne-1 year-1].
    The delimiter is comma.

    nEvents: number of events to simulate. 
    '''
    # Load event rate data
    EventRate_file = np.loadtxt(EventRateFile, skiprows=0,delimiter=',')
    energy, eventRate = EventRate_file[:,0], EventRate_file[:,1]
    
    # Calculate total rate and bin edges
    total_rate = np.trapz(eventRate, energy)
    bin_edges = np.zeros(len(energy)+1)
    bin_edges[1:-1] = (energy[1:] + energy[:-1])/2
    bin_edges[0] = energy[0] - (energy[1]-energy[0])/2
    bin_edges[-1] = energy[-1] + (energy[-1]-energy[-2])/2
    
    # Calculate probability for each bin
    bin_probs = eventRate * np.diff(bin_edges)
    bin_probs = bin_probs / np.sum(bin_probs)
    
    # Sample bin indices and uniformly sample within bins
    bin_indices = np.random.choice(len(energy), size=nEvents, p=bin_probs)
    sampled_energies = np.zeros(nEvents)
    for i in range(nEvents):
        bin_idx = bin_indices[i]
        sampled_energies[i] = np.random.uniform(bin_edges[bin_idx], bin_edges[bin_idx+1])

    
    weights = []

    nElectrons_NR = []
    nElectrons_NR_eff = []
    for E in sampled_energies:
        nElectrons_raw = LAr.energyToElectrons_NR(E)
        nElectrons_NR.append(nElectrons_raw)

        S2 = LAr.electronToS2(nElectrons_raw)
        nElectrons = S2/SingleElectronS2

        eff = LAr.S2OnlyEfficiency(nElectrons)
        nElectrons_NR_eff.append(nElectrons)

        if np.random.binomial(1, eff) == 1:
            weights.append(total_rate / nEvents)
        else:
            weights.append(0)

    weightsRaw = total_rate / nEvents * np.ones(nEvents)

    return nElectrons_NR, nElectrons_NR_eff, weightsRaw, weights




def simulate_LAr_Primary(EventRateFile, nEvents=1000000):
    '''
    Event rate file: output from MMA, first column is energy, second column is differential event rate [keV-1 tonne-1 year-1].
    The delimiter is comma.

    nEvents: number of events to simulate. 
    '''
    EventRate_file = np.loadtxt(EventRateFile, skiprows=0,delimiter=',')
    energy, eventRate = EventRate_file[:,0], EventRate_file[:,1]
    total_rate = np.trapz(eventRate, energy)
    sampled_energies = choices(energy, weights=eventRate, k=nEvents)

    

    nPrimary = []
    nRecombination = []
    nObserved = []
    
    for E in sampled_energies:
        primary, recombination = LAr.energyToElectrons_NR_full(E)
        nPrimary.append(primary)
        nRecombination.append(recombination)

        S2 = LAr.electronToS2(recombination)
        nElectrons = S2/SingleElectronS2
        nObserved.append(nElectrons)

    weights = total_rate / nEvents * np.ones(nEvents)

    return nPrimary, nRecombination, nObserved, sampled_energies, weights
