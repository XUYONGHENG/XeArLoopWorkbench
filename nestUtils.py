import nestpy
import scipy.stats as ss
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
import numpy as np
from math import erf

'''
The goal is to do as little repititon as possible. 
Since GetYields and GetQuanta do not depend on most detector parameters, we want to seperate the S1+S2 calculation from the yields calculation

Only repeat the yields/quanta calculations if the field changes

Only generate an energy spectrum once, and only generate a position spectrum once
'''
#Generate the quanta based on input energy and field only
def generate_quanta( NESTcalc, interaction, energy_array, field, density=2.9, SkewnessER=-999., 
                    width_params=[0.4,0.4,0.04,0.5,0.19,2.25, 0.0015, 0.046452, 0.205, 0.45, -0.2],
                    NRparams=[11., 1.1, 0.0480, -0.0533, 12.6, 0.3, 2., 0.3, 2., 0.5, 1., 1.], 
                    ERparams=[-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.]):
        quantas = []
        for i, energy in enumerate(energy_array):
                # if i % 10000 == 0:
                #         print(f'Processing energy {i} of {len(energy_array)}')
                yields = NESTcalc.GetYields( interaction, energy, drift_field = field, density=density, 
                                            nuisance_parameters=NRparams, ERYieldsParam=ERparams)
                quanta = NESTcalc.GetQuanta( yields, free_parameters=width_params, SkewnessER=SkewnessER )
                quantas.append( quanta )

        return np.array( quantas )



#Use the default NEST ER Weighting
def generate_quanta_ERweighted( NESTcalc, energy_array, field, density=2.9):
        quantas = []
        for energy in energy_array:
                yields = NESTcalc.GetYieldERWeighted( energy, drift_field = field, density=density )
                quanta = NESTcalc.GetQuanta( yields )
                quantas.append( quanta )

        return np.array( quantas )

# Create a custom beta+gamma weighting for ER, based on execNEST.cpp 
def generate_quanta_ERweightedCustom( NESTcalc, energy_array, field ):
        quantas = []
        yields = NESTcalc.GetYields( nestpy.INTERACTION_TYPE(8), 5., drift_field = field )

        #A, B, C, D, E, F = 0.5, 0.5, 1.1, -5., 1.01, 0.95 # LUX Run3
        A, B, C, D, E, F = 1., 0.55, -1.6, -1.0, 0.99, 1.04 # XENON10
        #A, B, C, D, E, F = -0.1, 0.5, 0.06, -0.6, 1.11, 0.95 # LUX C14
        for energy in energy_array:
                yieldsBeta = NESTcalc.GetYields( nestpy.INTERACTION_TYPE(8), energy, drift_field = field )
                yieldsGamma = NESTcalc.GetYields( nestpy.INTERACTION_TYPE(7), energy, drift_field = field )
                weightG = A + B * erf( C * (np.log( energy ) + D ) )         
                weightB = 1. - weightG

                yields.PhotonYield = weightG * yieldsGamma.PhotonYield + weightB * yieldsBeta.PhotonYield
                yields.ElectronYield = weightG * yieldsGamma.ElectronYield + weightB * yieldsBeta.ElectronYield
                yields.ExcitonRatio = weightG * yieldsGamma.ExcitonRatio + weightB * yieldsBeta.ExcitonRatio
                yields.Lindhard = weightG * yieldsGamma.Lindhard + weightB * yieldsBeta.Lindhard;
                yields.ElectricField = weightG * yieldsGamma.ElectricField + weightB * yieldsBeta.ElectricField;
                yields.DeltaT_Scint = weightG * yieldsGamma.DeltaT_Scint + weightB * yieldsBeta.DeltaT_Scint
                
                yields.PhotonYield *= E
                yields.ElectronYield *= F

                quanta = NESTcalc.GetQuanta( yields )
                quantas.append( quanta )
        
        return np.array( quantas )

# Make GetS1 usable with array/list inputs, while also simplifying the number of commands needed at runtime 
# (There are unneccessary C++ commands that get forced on us by the default bindings....)
        
def generate_S1( NESTcalc, interaction, energy_array, field, gen_quanta, xPos, yPos, zPos, driftVelocity ):         
        S1s = []
        for i in range(len(energy_array)):
                S1 = NESTcalc.GetS1( gen_quanta[i], xPos[i], yPos[i], zPos[i], xPos[i], yPos[i], zPos[i], driftVelocity, driftVelocity, interaction, 0, field, energy_array[i], nestpy.S1CalculationMode.Full, False, [0], [0])
                S1s.append( S1[5] )
        return np.array( S1s )

def generate_S1_highEnergy( NESTcalc, interaction, energy_array, field, gen_quanta, xPos, yPos, zPos, driftVelocity ):
        S1s = []
        spikes = []
        for i in range(0, len(energy_array)):
                #print(i)
                S1 = NESTcalc.GetS1( gen_quanta[i], xPos[i], yPos[i], zPos[i], xPos[i], yPos[i], zPos[i], driftVelocity, driftVelocity, interaction, 0, field, energy_array[i], nestpy.S1CalculationMode.Parametric, False, [0], [0])
                S1s.append( S1[5] )
                spikes.append( S1[7] )
        return np.array( S1s ), np.array( spikes )

# Do the same for GetS2...
def generate_S2( NESTcalc, interaction, energy_array, field, gen_quanta, xPos, yPos, zPos, driftVelocity, driftTimes ):
        g2Info = NESTcalc.CalculateG2(False)
        S2s = []
        for i in range(len(energy_array)):
                S2 = NESTcalc.GetS2( gen_quanta[i].electrons, xPos[i], yPos[i], zPos[i], xPos[i], yPos[i], zPos[i], driftTimes[i], driftVelocity, 0, field, nestpy.S2CalculationMode.Full, False, [0], [0], g2Info )
                S2s.append( S2[7] )
        return np.array( S2s )

#Do a hack of GetS2 to generate Single Electrons
#   Effectively, always make driftTime = 0, and extraction efficienty = 100%
#   And generate the S2 for a single electron
def generate_SE( NESTcalc, drift_field, xPos, yPos, zPos, driftVelocity ):
        g2Info = NESTcalc.CalculateG2(False)
        g2Info[1] = 1.00 # make extraction efficiency 1. This only effects the N_Extracted_Electrons calculation
        SEs = []
        for i in range(len(xPos)):
                SE = NESTcalc.GetS2( 1, xPos[i], yPos[i], zPos[i], xPos[i], yPos[i], zPos[i], 0., driftVelocity, 0, drift_field, nestpy.S2CalculationMode.Full, False, [0], [0], g2Info )
                SEs.append( abs( SE[7] ) ) # corrected SE area 
        return np.array( SEs )

#Define some functions just for ease of fitting with curve_fit(...)
def gaussian(x, mu, sigma, norm):
        return norm*np.exp( -( (x-mu)*(x-mu)/sigma/sigma )/2. )

def skew_gaussian(x, mu, sigma, norm, alpha):
        delta = alpha/np.sqrt(1. + alpha*alpha)
        skew_correction = 2*delta*delta/np.pi
        omega = sigma/np.sqrt( 1 - skew_correction )
        xi = mu - omega*np.sqrt(skew_correction)        
        return norm*ss.skewnorm.pdf(x, alpha, xi, omega)


def fit_band( x_array, y_array, xMin, xMax, xStep ):
            
        # Make bins for S1
        x_bin_edges = np.arange(xMin, xMax+xStep, xStep)
        x_bin_centers = (x_bin_edges[1:] + x_bin_edges[:-1]) / 2

        # Initially clean the data (remove S1s that are out of bounds)
        cleaning_mask = (x_array > xMin) * (x_array < xMax)
        x_array = x_array[cleaning_mask]
        y_array = y_array[cleaning_mask]

        means, mean_errs, widths, width_errs = [], [], [], []

        # Loop through the S1 bins
        for i in range(len(x_bin_centers)):

            # Make an S1 bin
            bin_mask = (x_array > x_bin_edges[i]) * (x_array < x_bin_edges[i+1])

            # Get the S2s falling inside this S1 bin
            y_array_in_bin = y_array[bin_mask]

            # Bin the S2s in preparation to fit
            #hist_y, y_bin_edges = np.histogram(y_array_in_bin, bins = 25, range = (y_array_in_bin.min(), y_array_in_bin.max()))
            #y_bin_centers = (y_bin_edges[1:] + y_bin_edges[:-1]) / 2

            # Fit the curve
            try:
                hist_y, y_bin_edges = np.histogram(y_array_in_bin, bins = 50, range = (y_array_in_bin.min()*0.95, y_array_in_bin.max()*1.05))
                y_bin_centers = (y_bin_edges[1:] + y_bin_edges[:-1]) / 2
                popt, pcov = curve_fit(gaussian, y_bin_centers, hist_y, p0 = [np.mean(y_array_in_bin), np.std(y_array_in_bin), len(y_array_in_bin)])
            except:
                print('Curve fit failed.')
                return None

            perr = np.sqrt(np.diag(pcov))

            means.append(popt[0])
            mean_errs.append(perr[0])
            widths.append(1.282 * popt[1]) # 90% CL
            width_errs.append(1.282 * perr[1])

        return np.array(x_bin_centers), np.array(means), np.array(mean_errs), np.array(widths), np.array(width_errs)


def get_deviation( data1, data2 ):
        #takes two data sets, and get the average absolute deviation between them
        if ( len(data1) != len(data2) ):
                print( "Data sets are not equal lengths! %i vs. %i" % (len(data1), len(data2)) )
                return 999
        deviation = 0.
        term1, term2 = 0., 0.
        count = 0
        for i in range(len(data1)):
                if data1[i] != 0.:
                        deviation += abs(data1[i] - data2[i])/data1[i]
                        term1 +=  abs(data1[i] - data2[i])/data1[i] #absolute relative deviation --> minimum is non-zero due to poisson fluctuations
                        term2 += (data1[i] - data2[i])/data1[i] #signed relative deviation --> goes to zero upon perfect fit
                        count += 1
        return term1/count + abs( term2 )
        return deviation#/count
                
def calculate_ChiSq( data1, data2, error1, error2, nFreeParams, power = 2):
                        
        chisq = sum(np.absolute(data1 - data2)**power / (np.absolute(error1)**power + np.absolute(error2)**power))
        count = len(data1)

        DoF = count - nFreeParams
        return chisq#/( DoF - 1 )

def calculate_SimpleChiSq( data1, data2 ):
        chisq = 0.
        if ( len(data1) != len(data2) ):
                print( "Data sets are not equal lengths! %i vs. %i" % (len(data1), len(data2)) )
                return 999
        count = 0
        for i in range(len(data1)):
                if data1[i] > 0:
                        chisq +=  pow( (data1[i] - data2[i])/(data1[i]), 2. )
                        count += 1

        return chisq/count


