"""
Created on Sat Oct 23 19:45:03 2016

@author: Pavel
"""
"""
Heterogeneous Consumers with Financial Frictions 

This program applies the preference shock model with kinks to analyze the 
effects of financial frictions in an economy populated with heterogeneous 
consumers (this is modeled through a varying spread between the saving and 
borrowing rates faced by different consumers). The analysis considers consumers
that are subject to income shocks and preference shocks.
"""


import matplotlib.pyplot as plt # used to plot consumption functions
#import pylab as plt # used to plot the saving rate
import numpy as np

from copy import deepcopy
from HARKutilities import plotFuncs
#from time import clock
import ConsumerParameters
#import SetupParamsCSTW as Params
from ConsIndShockModel import KinkedRconsumerType
from ConsPrefShockModel import KinkyPrefConsumerType
from HARKutilities import approxUniform, getLorenzShares
#from HARKsimulation import drawDiscrete
#from cstwKinkedR import cstwKinkedRagent 


#mystr = lambda number : "{:.4f}".format(number)
#'''
#Truncates a float at exactly four decimal places when displaying as a string.
#'''

###############################################################################

# =================================================================
# ====== Function to assign heterogeneity in rates ================
# =================================================================

def assignRdistribution(type_list,R_list,isRboro):
    '''
    Assigns the interest rate values in values_list to the types in type_list.  If
    there is heterogeneity beyond the interest rate, then the same value is
    assigned to consecutive types (that is why the function uses two while loops).
    It allows to assign heterogeneity in the interest rate on saving and on debt.
    
    Parameters
    ----------
    type_list : [agent]
        The list of types that should be assigned the different values.
    R_list : [float] or np.array
        List of values to assign to the types.
    isRboro : boolean
        Assigns the values in R_list to the interest rate for borrowing when 
        True, to the interest rate on savings when False        
        
    Returns
    -------
    none
    '''
    R_count = len(R_list)
    type_N = len(type_list)/R_count
    j = 0
    b = 0
    while j < len(type_list):
        t = 0
        while t < type_N:
            if isRboro:
                type_list[j](Rboro = R_list[b])
            else:
                type_list[j](Rsave = R_list[b])
            t += 1
            j += 1
        b += 1           

###############################################################################
"""
Define the type of consumers to be used
"""
KinkyExample = KinkedRconsumerType(**ConsumerParameters.init_kinked_R)
KinkyExample.cycles = 0 # Infinite horizon

KinkyPrefExample = KinkyPrefConsumerType(**ConsumerParameters.init_kinky_pref)
KinkyPrefExample.cycles = 0 # Infinite horizon

"""
Add in ex-ante heterogeneity in consumers' borrowing and saving rates
"""

# First create the desired number of consumer types

num_consumer_types   = 7       # declare the number of types we want
SpreadRconsumerTypes = []       # initialize an empty list for kinky
SpreadRwPrefConsumerTypes = []  # initialize an empty list for kinky + preference

for nn in range(num_consumer_types):
    # Now create the types, and append them to both lists of consumer types
    newType1 = deepcopy(KinkyExample)   
    newType2 = deepcopy(KinkyPrefExample)
    SpreadRconsumerTypes.append(newType1)    
    SpreadRwPrefConsumerTypes.append(newType2)

## Now, generate the desired ex-ante heterogeneity, by giving the different consumer types
## each with their own borrowing and saving rate

# First, decide the borrowing and saving rates to assign
bottomRboro = 1.02
topRboro    = 1.2
Rboro_list  = approxUniform(N=num_consumer_types,bot=bottomRboro,top=topRboro)[1]   # only take values, not probs
Rboro_list.sort() # sort the list from smaller to larger saving rate


#bottomRsave = 1.01
#topRsave    = 1.2
#Rsave_list  = approxUniform(N=num_consumer_types,bot=bottomRsave,top=topRsave)[1]   # only take values, not probs
#Rsave_list.sort() # sort the list from smaller to larger saving rate
#
#Rboro_list = []
#bottomSpread = .01
#topSpread    = .1
#Spread_list  = approxUniform(N=num_consumer_types,bot=bottomSpread,top=topSpread)[1] # only take values, not probs
#for nn in range(num_consumer_types):
#    Rboro_list.append(Rsave_list[nn] + Spread_list[nn])

# Now, assign the borrowing and saving rates we want to both lists of consumer types
#assignRvalues(SpreadRconsumerTypes,Rsave_list,False)               # assigns Rsave
assignRdistribution(SpreadRconsumerTypes,Rboro_list,isRboro=True)   # assigns Rboro
#assignRvalues(SpreadRwPrefConsumerTypes,Rsave_list,False)          # assigns Rsave
assignRdistribution(SpreadRwPrefConsumerTypes,Rboro_list,isRboro=True)    # assigns Rboro

###############################################################################

def calcNatlSavingRate(type_list, isPrefType, RNG_seed = 0):
    """
    This function performs the experiment: What happens to the path for the 
    national saving rate when consumers face different spreads in interest rates?
 
    Parameters
    ----------
    type_list : list
        List of types that will be solved and simulated
    isPrefType : boolean
        True if the consumer is of the kinky + preference type
    RNG_seed : integer
        to seed the random number generator for simulations.  This useful
          because we are going to run this function for different consumer types,
          and we may not necessarily want the simulated agents in each run to experience
          the same (normalized) shocks.
        
    Returns
    -------
    [NatlSavingRateRsave,NatlSavingRateRboro] : list
        Evolution of the national saving rate using the rate on savings and the rate on borrowing    
    """

    # To calculate the national saving rate, we need national income and national consumption
    # To get those, we are going to start national income and consumption at 0, and then
    # loop through each agent type and see how much they contribute to income and consumption.
    NatlIncomeRsave = 0.
    NatlIncomeRboro = 0.
    NatlCons   = 0.
    
    numIteration = 0 # count the number of iteration in the loop

    for SpreadConsumerType in type_list:
        ### For each consumer type (i.e. each pair of borrowing and saving rates),
        ###  calculate total income and consumption

        # First give each ConsumerType their own random number seed
        RNG_seed += 19
        SpreadConsumerType.seed  = RNG_seed

        # Solve the problem for this ChineseConsumerTypeNew
        SpreadConsumerType.solve()

        #if numIteration == 0:
        if isPrefType:
            # Plot the consumption function at each discrete shock
            m = np.linspace(SpreadConsumerType.solution[0].mNrmMin,5,200)
            print('Consumption functions at each discrete shock:')
            fig = plt.figure()            
            for j in range(SpreadConsumerType.PrefShkDstn[0][1].size):
                PrefShk = SpreadConsumerType.PrefShkDstn[0][1][j]
                c = SpreadConsumerType.solution[0].cFunc(m,PrefShk*np.ones_like(m))
                plt.plot(m,c)
            ax = fig.add_subplot(111)            
            #x = np.linspace(*ax.get_xlim(),**ax.get_ylim())
            ax.plot([max(ax.get_xlim()[0], ax.get_ylim()[0]), 
         min(ax.get_xlim()[1], ax.get_ylim()[1])],
        [max(ax.get_xlim()[0], ax.get_ylim()[0]), 
         min(ax.get_xlim()[1], ax.get_ylim()[1])], ls=":", color = '0.2')
            plt.show()
        else:
            SpreadConsumerType.unpackcFunc()
            print('Kinky consumption function:')
            fig = plt.figure()
            SpreadConsumerType.timeFwd()       
#            plotFuncs(SpreadConsumerType.cFunc[0],SpreadConsumerType.solution[0].mNrmMin,5)   
#            plt.show()
#            ax = fig.add_subplot(111)            
#            #x = np.linspace(*ax.get_xlim())            
#            ax.plot([max(ax.get_xlim()[0], ax.get_ylim()[0]), 
#                     min(ax.get_xlim()[1], ax.get_ylim()[1])],
#                    [max(ax.get_xlim()[0], ax.get_ylim()[0]), 
#                    min(ax.get_xlim()[1], ax.get_ylim()[1])], ls="--", color = '0.2')
#            plotFuncs(SpreadConsumerType.cFunc[0],SpreadConsumerType.solution[0].mNrmMin,5)   
#            plt.show()
            
            function = SpreadConsumerType.cFunc[0]
            
            bottom = SpreadConsumerType.solution[0].mNrmMin
            top = 5
            N=1000
            step = (top-bottom)/N
            x = np.arange(bottom,top,step)
            y = function(x)
            plt.plot(x,y)
            plt.xlim([bottom, top])
            ax = fig.add_subplot(111)            
            #x = np.linspace(*ax.get_xlim())            
            ax.plot([max(ax.get_xlim()[0], ax.get_ylim()[0]), 
                     min(ax.get_xlim()[1], ax.get_ylim()[1])],
                    [max(ax.get_xlim()[0], ax.get_ylim()[0]), 
                    min(ax.get_xlim()[1], ax.get_ylim()[1])], ls=":", color = '0.2')
            plt.show()
            
            
        """
        # Plot of consumption function over time
        topp = len(SpreadConsumerType.cHist)
        bottomm = 0
        stepp = 1
        
        xx = np.arange(bottomm,topp,stepp)
        yy = [0] * 120
        aa = 0
        
        for k in range(len(SpreadConsumerType.cHist)):
            aa = (SpreadConsumerType.cHist[k,1])
            yy[k] = aa
        
        plt.plot(xx,yy)
        plt.show()
        """
        """
        Now we are ready to simulate.
        """
        SpreadConsumerType.sim_periods = 120
        SpreadConsumerType.makeIncShkHist() # create the history of income shocks
        if isPrefType:                      # create the history of preference shocks
            SpreadConsumerType.makePrefShkHist()
        SpreadConsumerType.initializeSim()  # get ready to simulate everything else
        SpreadConsumerType.simConsHistory() # simulate everything else
        
        # Now, get the aggregate income and consumption of this ConsumerType
        IncomeOfThisConsumerTypeRsave = np.sum((SpreadConsumerType.aHist * SpreadConsumerType.pHist*
                                          (SpreadConsumerType.Rsave - 1.)) + 
                                           SpreadConsumerType.pHist, axis=1)
        
        IncomeOfThisConsumerTypeRboro = np.sum((SpreadConsumerType.aHist * SpreadConsumerType.pHist*
                                          (SpreadConsumerType.Rboro - 1.)) + 
                                           SpreadConsumerType.pHist, axis=1)        
        
        ConsOfThisConsumerType = np.sum(SpreadConsumerType.cHist * SpreadConsumerType.pHist,
                                        axis=1)
        
        # Add the income and consumption of this ConsumerType to national income and consumption
        NatlIncomeRsave     += IncomeOfThisConsumerTypeRsave
        NatlIncomeRboro     += IncomeOfThisConsumerTypeRboro
        NatlCons       += ConsOfThisConsumerType

        numIteration += 1
        
    # After looping through all the ConsumerTypes, calculate and return the path of the national 
    # saving rate
    NatlSavingRateRsave = (NatlIncomeRsave - NatlCons)/NatlIncomeRsave
    NatlSavingRateRboro = (NatlIncomeRboro - NatlCons)/NatlIncomeRboro

    return [NatlSavingRateRsave,NatlSavingRateRboro]


def plotNatlSavingRate(NatlSavingsRates,labels):
    """
    This function the path for the national saving rate for kinky and 
    kinky + preference consumers
   
    Parameters
    ----------
    NatlSavingsRates : list
        Evolutions of the national saving rate
    labels : list of strings
        Labels of the lines to be plotted
        
    Returns
    -------
    none    
    """
    
    plt.ylabel('Natl Savings Rate')
    plt.xlabel('Quarters')
    plt.plot(quarters_to_plot,NatlSavingsRates[0],label=labels[0])
    plt.plot(quarters_to_plot,NatlSavingsRates[1],label=labels[1])
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.) #put the legend on top
           
####################################################################################################
"""
Now we can use the function we just defined to calculate the path of the 
national saving rate. We are going to graph this path for kinky and 
kinky+preference consumer types.
"""

# Declare the quarters we want to plot results for
quarters_to_plot = np.arange(0,120,1)

# Create a list to hold the paths of the national saving rate

NatlSavingsRatesRsave = []
NatlSavingsRatesRboro = []

#[-160 - quarters_before_reform_to_plot :])
NatlSavingsRates1 = calcNatlSavingRate(SpreadRconsumerTypes,False,RNG_seed = 1)
NatlSavingsRates2 = calcNatlSavingRate(SpreadRwPrefConsumerTypes,True,RNG_seed = 2)
NatlSavingsRatesRsave.append(NatlSavingsRates1[0]) # Kinky consumer
NatlSavingsRatesRsave.append(NatlSavingsRates2[0]) # Kinky + Preference consumer
NatlSavingsRatesRboro.append(NatlSavingsRates1[1]) # Kinky consumer
NatlSavingsRatesRboro.append(NatlSavingsRates2[1]) # Kinky + Preference consumer
#NatlSavingsRates.append(calcNatlSavingRate(SpreadRconsumerTypes,False,RNG_seed = 1)) 
#NatlSavingsRates.append(calcNatlSavingRate(SpreadRwPrefConsumerTypes,True,RNG_seed = 2))

# We've calculated the path of the national saving rate as we wanted
# All that's left is to graph the results!
labels = ['Kinky', 'Kinky + Preference']
plt.figure(0)
#plt.subplot(211)
plotNatlSavingRate(NatlSavingsRatesRsave, labels)
plt.figure(1)
#plt.subplot(212)
plotNatlSavingRate(NatlSavingsRatesRboro, labels)
labels = ['Rate on savings', 'Rate on borrowing']
plt.figure(2)
#plt.subplot(221)
plotNatlSavingRate([NatlSavingsRatesRsave[0],NatlSavingsRatesRboro[0]],labels) # Kinky consumer
plt.figure(4)
#plt.subplot(222)
plotNatlSavingRate([NatlSavingsRatesRsave[1],NatlSavingsRatesRboro[1]],labels) # Kinky + Preference consumer

###############################################################################
  
"""     
# Plot of consumption function over time
topp = len(KinkyExample.cHist)
bottomm = 0
stepp = 1

xx = np.arange(bottomm,topp,stepp)
yy = [0] * 120
aa = 0

for k in range(len(KinkyExample.cHist)):
    aa = (KinkyExample.cHist[k,1]) # => cHist mitrix of 120 x 1 i.e. of simulations x agents
    yy[k] = aa

plt.plot(xx,yy)
plt.show()
"""

#
##==============================================================
## =====Estimation of cstw model with kinky consumers ==========
##==============================================================
#
#Params.do_lifecycle = False            # Perpetual youth 
#Params.do_liquid = True     # Matches liquid assets data when True, net worth data when False
#Params.do_tractable = False 
#Params.do_agg_shocks = True           # Aggregate shocks model (for now) only supports IndShockConsumerType
#Params.run_estimation = True
#Params.percentiles_to_match = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # Which points of the Lorenz curve to match in Rboro-dist (must be in (0,1))
#
#
## =================================================================
## ====== Make the list of consumer types for estimation ===========
##==================================================================
#
## Set target Lorenz points and K/Y ratio (MOVE THIS TO SetupParams)
#if Params.do_liquid:
#        #!!! Temporal in order to allow for variable number of percentiles to match
#    lorenz_target = getLorenzShares(Params.SCF_wealth,weights=Params.SCF_weights,percentiles=Params.percentiles_to_match) 
#    #lorenz_target = np.array([0.0, 0.004, 0.025,0.117])
#    #KY_target = 6.60
#    
#    # Proportion of households having exactly zero liquid assets
#    pctHHassets_target = float(np.sum(i == 0 for i in Params.SCF_wealth))/len(Params.SCF_wealth)
#    # Alternatively, we can use the line below if we want to include HH with ALMOST zero liquid assets    
#    # pctHHassets_target = float(np.sum(100 <= i <= 100 for i in Params.SCF_wealth))/len(Params.SCF_wealth)
#    
#   
## Make a vector of initial wealth-to-permanent income ratios
#a_init = drawDiscrete(N=Params.sim_pop_size,P=Params.a0_probs,X=Params.a0_values,seed=Params.a0_seed)
#                                         
## Make the list of types for this run
## Make the base infinite horizon type and assign income shocks
#InfiniteType = cstwKinkedRagent(**Params.init_kinked_R)
#InfiniteType.tolerance = 0.0001
#InfiniteType.a_init = 0*np.ones_like(a_init)
#
## Make histories of permanent income levels for the infinite horizon type
#p_init_base = np.ones(Params.sim_pop_size,dtype=float)
#InfiniteType.p_init = p_init_base
#
#short_type_list = [InfiniteType]
#spec_add = 'IH'
#
#
## Expand the estimation type list if doing Rboro-dist
#if Params.do_Rboro_dist:
#    long_type_list = []
#    for j in range(Params.pref_type_count):
#        long_type_list += deepcopy(short_type_list)
#    est_type_list = long_type_list
#else:
#    est_type_list = short_type_list
#
#if Params.do_liquid:
#    wealth_measure = 'Liquid'
#else:
#    wealth_measure = 'NetWorth'
#
#
## =================================================================
## ====== Define estimation objectives =============================
##==================================================================
#
## Set commands for the Rboro-point estimation
#Rboro_point_commands = ['solve()','unpackcFunc()','timeFwd()','simulateCSTW()']
#    
## Make the objective function for the Rboro-point estimation
#RboroPointObjective = lambda Rboro : simulatePctHHassetsDifference(Rboro,
#                                                             nabla=0,
#                                                             N=1,
#                                                             type_list=est_type_list,
#                                                             weights=Params.age_weight_all,
#                                                             target=pctHHassets_target)
#                                                             
## Make the objective function for the Rboro-dist estimation
#def RboroDistObjective(nabla):   #!!! Finds the nabla that minimizes the distance with the target
#    # Make the "intermediate objective function" for the Rboro-dist estimation
#    #print('Trying nabla=' + str(nabla))
#    intermediateObjective = lambda Rboro : simulatePctHHassetsDifference(Rboro,
#                                                             nabla=nabla,
#                                                             N=Params.pref_type_count,
#                                                             type_list=est_type_list,
#                                                             weights=Params.age_weight_all,
#                                                             target=pctHHassets_target)
#
#    top = 1.2
#    Rboro_new = brentq(intermediateObjective,1.02,top,xtol=10**(-8)) 
#    N=Params.pref_type_count
#    sim_wealth = (np.vstack((this_type.W_history for this_type in est_type_list))).flatten()
#    sim_weights = np.tile(np.repeat(Params.age_weight_all,Params.sim_pop_size),N)
#    my_diff = calculateLorenzDifference(sim_wealth,sim_weights,Params.percentiles_to_match,lorenz_target)
#    print('Rboro=' + str(Rboro_new) + ', nabla=' + str(nabla) + ', diff=' + str(my_diff))
#    if my_diff < Params.diff_save:
#        Params.Rboro_save = Rboro_new
#    return my_diff
#
#
#
## =================================================================
## ========= Estimating the model ==================================
##==================================================================
#
#if Params.run_estimation:
#    # Estimate the model and time it
#    t_start = time()
#    if Params.do_Rboro_dist:
#        bracket = (0,0.15)
#        # bracket = (0,0.015) #!!! large nablas break IH version in cstw, but in kinked?
#        nabla = golden(RboroDistObjective,brack=bracket,tol=10**(-4))        
#        Rboro = Params.Rboro_save
#        spec_name = spec_add + 'RboroDist' + wealth_measure
#    else:
#        nabla = 0 #!!! Used when call makeCSTWresults
#        bot = 1.02
#        top = 1.2
#        Rboro = brentq(RboroPointObjective,bot,top,xtol=10**(-8))
#        spec_name = spec_add + 'RboroPoint' + wealth_measure
#    t_end = time()
#    print('Estimate is Rboro=' + str(Rboro) + ', nabla=' + str(nabla) + ', took ' + str(t_end-t_start) + ' seconds.')
#    #spec_name=None
#    makeCSTWresults(Rboro,nabla,save_name=None) #spec_name) in order to save figures

