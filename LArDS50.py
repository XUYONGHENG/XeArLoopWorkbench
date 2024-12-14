import numpy as np
import os 

Wq_Ar = 0.0195 # keV/quantum

driftField = 200 # V/cm
CBox_ER = 9.2 # V/cm

gamma_ER_Ar = CBox_ER/driftField

p0 = 0.11
p1 = 1.71

# ER yield models


def primary_ionization_yield(Eer):
    # Eer is the energy of the primary particle, in keV
    # Return the number of ion/electron pairs, before recombination
    return 52.7*Eer

def recombination_probability(Ni): 
    # Ni is the primary ion/electron pairs number, a dimensionless quantity, obtained by yields
    # gamma_ER_Ar is the recombination coefficient
    return 1 - (1/(gamma_ER_Ar*Ni)*np.log(1 + gamma_ER_Ar*Ni))


def Ar39_modification_factor(Eer):
    return gamma_ER_Ar*p0*Eer**p1


def totalChargeYieldExpectationRaw(Eer):
    yield_Ar39 = primary_ionization_yield(Eer)
    prob_survival = 1 - recombination_probability(yield_Ar39)
    return yield_Ar39*prob_survival / Eer



def totalChargeYieldExpectation(Eer):
    yield_Ar39 = primary_ionization_yield(Eer) * (1 + Ar39_modification_factor(Eer))
    prob_survival = 1 - recombination_probability(primary_ionization_yield(Eer))
    return yield_Ar39*prob_survival / Eer


def energyToElectrons_ER(Eer, rng=np.random.default_rng()):
    # Calculate initial quanta before recombination
    initial_quanta = primary_ionization_yield(Eer) * (1 + Ar39_modification_factor(Eer))
    # Apply binomial fluctuation in recombination
    prob_survival = 1 - recombination_probability(primary_ionization_yield(Eer))
    return rng.binomial(n=int(initial_quanta), p=prob_survival)


## NR yield models

beta_NR = 6800
fZ = 0.953

def epsilon_NR(Enr):
    return 0.0135*Enr

def stoppingPowerE(epsilon):
    return 0.145*np.sqrt(epsilon)

def stoppingPowerN(epsilon):
    return np.log(1 + 1.1383 * fZ * epsilon) / (2*(fZ*epsilon + 0.01321*(fZ*epsilon)**0.21226 + 0.19593*(fZ*epsilon)**0.5))

def primaryIonNumberNR(Enr):
    return beta_NR * (epsilon_NR(Enr) * stoppingPowerE(epsilon_NR(Enr)))/(stoppingPowerE(epsilon_NR(Enr)) + stoppingPowerN(epsilon_NR(Enr)))

def postRecombinationPrimaryIonNumberNR(Enr):
    prob_survival = 1 - recombination_probability(primaryIonNumberNR(Enr))
    return primaryIonNumberNR(Enr) * prob_survival


def energyToElectrons_NR(Enr, rng=np.random.default_rng()):
    # Calculate initial quanta before recombination for NR
    initial_quanta = primaryIonNumberNR(Enr)
    # Apply binomial fluctuation in recombination
    prob_survival = 1 - recombination_probability(primaryIonNumberNR(Enr))
    return rng.binomial(n=int(initial_quanta), p=prob_survival)


def energyToElectrons_NR_full(Enr, rng=np.random.default_rng()):
    # Calculate initial quanta before recombination for NR
    initial_quanta = primaryIonNumberNR(Enr)
    # Apply binomial fluctuation in recombination
    prob_survival = 1 - recombination_probability(primaryIonNumberNR(Enr))
    return initial_quanta, rng.binomial(n=int(initial_quanta), p=prob_survival)



def electronToS2(nElectrons, rng=np.random.default_rng()):
    # For each electron, generate S2 signal with mean=23 and std=1, then sum
    if nElectrons <= 0:
        return 0
    return np.sum(rng.normal(loc=23, scale=6.21, size=nElectrons))

def S2OnlyEfficiency(nElectrons):
    if nElectrons <= 2:
        return 0
    return 0.4