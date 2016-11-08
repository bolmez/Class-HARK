# -*- coding: utf-8 -*-
"""
Created on Sat Oct  22 23:22:53 2016

@author: Pavel
"""

###############################################################################

if __name__ == '__main__':
    import ConsumerParameters as Params
    import matplotlib.pyplot as plt    
    import numpy as np
    from copy import deepcopy
    from HARKutilities import plotFuncs
    from HARKutilities import approxUniform
    from ConsIndShockModel import KinkedRconsumerType
    from ConsPrefShockModel import KinkyPrefConsumerType
    from time import clock
    mystr = lambda number : "{:.4f}".format(number)

    do_simulation           = True
    
###############################################################################
#### Individual Shocks Model
###############################################################################     

    # Make and solve an agent with a kinky interest rate
    KinkyExample = KinkedRconsumerType(**Params.init_kinked_R)
    KinkyExample.cycles = 0 # Make the Example infinite horizon
    
    start_time = clock()
    KinkyExample.solve()
    end_time = clock()
    print('Solving a kinky consumer took ' + mystr(end_time-start_time) + ' seconds.')
    KinkyExample.unpackcFunc()
    print('Kinky consumption function:')
    KinkyExample.timeFwd()
    plotFuncs(KinkyExample.cFunc[0],KinkyExample.solution[0].mNrmMin,5)

    if do_simulation:
        KinkyExample.sim_periods = 120
        KinkyExample.makeIncShkHist()
        KinkyExample.initializeSim()
        KinkyExample.simConsHistory()

##### #Plot the consumption function over time
    topp = len(KinkyExample.cHist)
    bottomm = 0
    stepp = 1

    xx = np.arange(bottomm,topp,stepp)
    yy = [0] * 120
    aa = 0
    
    for k in range(len(KinkyExample.cHist)):
        aa= (KinkyExample.cHist[k,1])
        yy[k] = aa
    plt.plot(xx,yy)
    plt.show()

###############################################################################        
##### Preference Shocks Model
###############################################################################
        
    # Make and solve a "kinky preferece" consumer, whose model combines KinkedR and PrefShock
    KinkyPrefExample = KinkyPrefConsumerType(**Params.init_kinky_pref)
    KinkyPrefExample.cycles = 0 # Infinite horizon
    
    t_start = clock()
    KinkyPrefExample.solve()
    t_end = clock()
    print('Solving a kinky preference consumer took ' + str(t_end-t_start) + ' seconds.')
    
    # Plot the consumption function at each discrete shock
    m = np.linspace(KinkyPrefExample.solution[0].mNrmMin,5,200)
    print('Consumption functions at each discrete shock:')
    for j in range(KinkyPrefExample.PrefShkDstn[0][1].size):
        PrefShk = KinkyPrefExample.PrefShkDstn[0][1][j]
        c = KinkyPrefExample.solution[0].cFunc(m,PrefShk*np.ones_like(m))
        plt.plot(m,c)
    plt.show()
    
    print('Consumption function (and MPC) when shock=1:')
    c = KinkyPrefExample.solution[0].cFunc(m,np.ones_like(m))
    k = KinkyPrefExample.solution[0].cFunc.derivativeX(m,np.ones_like(m))
    plt.plot(m,c)
    plt.plot(m,k)
    plt.show()
    
    if KinkyPrefExample.vFuncBool:
        print('Value function (unconditional on shock):')
        plotFuncs(KinkyPrefExample.solution[0].vFunc,KinkyPrefExample.solution[0].mNrmMin+0.5,5)
        
    # Test the simulator for the kinky preference class
    if do_simulation:
        KinkyPrefExample.sim_periods = 120
        KinkyPrefExample.makeIncShkHist()
        KinkyPrefExample.makePrefShkHist()
        KinkyPrefExample.initializeSim()
        KinkyPrefExample.simConsHistory()
        
####################################################################################################
"""
Now, add in ex-ante heterogeneity in consumers' borrowing and saving rates
"""

# To create ex-ante heterogeneity in the borrowing and saving rates, first create
# the desired number of consumer types

num_consumer_types   = 10 # declare the number of types we want
KinkedRprefConsumerTypes = [] # initialize an empty list

for nn in range(num_consumer_types):
    # Now create the types, and append them to the list KinkedRprefConsumerTypes
    newType = deepcopy(KinkyExample)    
    KinkedRprefConsumerTypes.append(newType)

## Now, generate the desired ex-ante heterogeneity, by giving the different consumer types
## each with their own borrowing and saving rates

# First, decide the discount factors to assign

bottomRboro = 1.20           # Interest factor on assets when borrowing, a < 0
topRsave = 1.02           # Interest factor on assets when saving, a > 0
DiscFac_list  = approxUniform(N=num_consumer_types,bot=bottomRboro,top=topRsave)[1]

# Now, assign the discount factors we want to the KinkedRprefConsumerTypes
## ???? cstwMPC.assignBetaDistribution(KinkedRprefConsumerTypes,DiscFac_list)

###############################################################################






