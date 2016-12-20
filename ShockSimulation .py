#reset before you start!
#Generate a panel data from the income process
import math 

import sys 
import os

sys.path.insert(0,'../')

import matplotlib
import time
import numpy.matlib
import numpy as np
from scipy.optimize import minimize
import scipy.optimize
from scipy.optimize import fmin
#from scipy.optimize import maximize

import scipy.stats as stats
import random
import matplotlib.pyplot as plt
from astropy.table import Table, Column
from random import gauss

global pers 
global NumberOfPeriods


# Basic setup - NO DATA ONLY RANDOM
NumberOfHouseholds = 10000
NumberOfPeriods = 40
StartAge = 26
EndAge = StartAge + NumberOfPeriods -1
Heterogeneity = 1
nparms = 3+ NumberOfPeriods+(NumberOfPeriods-2)  

var_alpha  = 0.03; #random generation of income
#ALPHA - time varying coefficient
alpha1 = np.mean(np.random.randn(1,NumberOfHouseholds))
alpha = math.sqrt(var_alpha)*(np.random.randn(1,NumberOfHouseholds)-alpha1)


##BETA - time varying coefficient
if Heterogeneity == 1:
    var_beta = 0.004
    beta1 = np.mean(np.random.randn(1,NumberOfHouseholds))
    beta = math.sqrt(var_beta)*(np.random.randn(1,NumberOfHouseholds)-beta1)
else:
    beta = np.zeros(1,NumberOfHouseholds)
    
#Permanent component of the income shock - omega in the paper
var_psi = 0.002
atf = np.mean(beta)
psi = math.sqrt(var_psi)*(np.random.randn(NumberOfPeriods+1, NumberOfHouseholds)-beta1) 
#sum of persistent shock and random walk


#Transitory component of the income shock - nu in the paper

var_tran = 0.007 #this variance may depend on the industry or group!
tran = math.sqrt(var_tran)*(np.random.randn(NumberOfPeriods+1, NumberOfHouseholds)-beta1) + abs(atf)*(np.random.randn(NumberOfPeriods+1, NumberOfHouseholds)-beta1)
#transitory component is iid

phi = 0.05
theta = 0.02

ALL = psi + tran #total shock as the sum of persistent and the transitory

PermanentComponent = np.zeros((NumberOfPeriods+1, NumberOfHouseholds))
TransitoryComponent = np.zeros((NumberOfPeriods+1, NumberOfHouseholds))
ALLComponent = np.zeros((NumberOfPeriods+1, NumberOfHouseholds))
#rows as periods, columns as households

for i in range(0,NumberOfPeriods,1):
    PermanentComponent[i+1][:] = psi[i+1][:] + PermanentComponent[i][:]
    TransitoryComponent[i+1][:] = phi*TransitoryComponent[i][:] + tran[i+1][:] + theta*tran[i][:]
    ALLComponent[i+1][:] = PermanentComponent[i+1][:] + TransitoryComponent[i+1][:]

agemat = np.zeros((EndAge+2-StartAge,1))
k =0
for i in range(StartAge-1,EndAge):
    agemat[k][0] = i
    k = k +1


aaa= np.array(np.matlib.repmat(beta,NumberOfPeriods+1,1))
bbb = np.array(agemat*np.ones((1,NumberOfHouseholds)))
HeterogeneityComponent = np.matlib.repmat(alpha,NumberOfPeriods+1,1) + aaa*bbb
   
SimulatedResidualLogIncome =  HeterogeneityComponent + PermanentComponent + TransitoryComponent 

#Only for the TRANS Shocks
SimulatedTrans = HeterogeneityComponent + TransitoryComponent

res = (SimulatedResidualLogIncome[1:][:]-SimulatedResidualLogIncome[0:-1][:]).transpose()
resTRA = res = (SimulatedTrans[1:][:]-SimulatedTrans[0:-1][:]).transpose()

#until to the secont to last element
pers = NumberOfPeriods;


#################################################################################
#autocovariancemat()


OMEGA=np.zeros((pers,pers)) 
tOMEGA=np.zeros((pers,pers))
    # return the number of rows (heads) in matrix
i=res.shape
ii = np.array(i)
l1=ii[0]
    # denominator, counting the number of heads contributing to
    # autocovariances

denom=np.zeros((l1,pers*(pers+1)/2)) 
Tdenom=np.zeros((l1,pers*(pers+1)/2))
    # l1 is the total number of heads in the sample;
    #pers is the number of periods; 
    #for each head there are pers*(pers+1)/2 autocovariances

lag=np.zeros((1,pers*(pers+1)/2))
a=np.zeros((1,pers*(pers+1)/2))
q=np.zeros((1,pers*(pers+1)/2))    
    #time for some loops within loops!        

j=0    
for i in range(j+1,pers):
    for j in range(i-1,0,-1):
        lag[0][(i*(i+1)/2-j)-1]=j      


for i in range(0,l1,1):    # for each household head
    OMEGA_i=(np.array(res[i][:])[np.newaxis]).transpose()*res[i][:] 
    B = OMEGA_i.shape
    B0 = list(B)
    B1 = B0[0] 
    B2 = B0[1]
    BM= np.ones((B1,B2)) 
    UBM = np.triu(BM) #triu
    
    w= np.array(OMEGA_i[np.triu_indices(B1)]) 
    
    for j in range(0,pers*(pers+1)/2,1):
        #print "t"
        if w[j] != 0.00: 
            denom[i][j] = 1
        elif w[j] == 0.00:
            denom[i][j] = 0              
    OMEGA=OMEGA+OMEGA_i

#observing for the transitory shocks    
    OMEGA_t=(np.array(resTRA[i][:])[np.newaxis]).transpose()*resTRA[i][:]        
    T = OMEGA_t.shape
    T0 = list(T)
    T1 = T0[0] 
    T2 = T0[1]
    TM= np.ones((T1,T2)) 
    UTM = np.triu(TM)
    w= np.array(OMEGA_t[np.triu_indices(T1)]) 
    
    for j in range(0,pers*(pers+1)/2,1):
        #print "t"
        if w[j] != 0.00: #
            Tdenom[i][j] = 1
        elif w[j] == 0.00:
            Tdenom[i][j] = 0              
    OMEGAT=(OMEGA)+OMEGA_t

denom1=sum(denom)  
Tdenom1=sum(Tdenom)
    
OMEGA10=np.zeros((pers,pers))   

for i in range(0,pers,1): 
    for j in range(0,pers,1): 
        if j-i>4: 
            OMEGA10[i][j]=0 
        elif j-i<=4: 
            OMEGA10[i][j]=OMEGAT[i][j]        
C = OMEGA.shape
C0 = list(C)
C1 = C0[0] 
C2 = C0[1]
CM= np.ones((C1,C2))
UCM = np.triu(CM)
ww= np.array(OMEGA[np.triu_indices(C1)])[np.newaxis]/NumberOfHouseholds
wt= np.array(OMEGA10[np.triu_indices(C1)])[np.newaxis]/NumberOfHouseholds
    #OMEGA is a vector of the mean  autocovariances. 
    #OMEGA =[ (yr1, yr1), (yr1, yr2), (yr2, yr2), (yr1, yr3), (yr2, yr3), (yr3, yr3),...]
    #where (yr_i, yr_j) is the mean autocovariance between year i and year j


avarres=np.zeros((1, pers)) 
avarrest=np.zeros((1, pers)) #mean of autocovariances at lags 0-(pers-1)
    #FIGURE: Autocovariancefunction by lag
for i in range(0,pers-1,1):
    a[0][:] = lag[0][:]
    m1 = a.shape
    m2 = list(m1) 
    m4 = m2[1]    
    for j in range(0,m4,1):   
        if a[0][j]==i:
           a[0][j]=1 
        else:
           a[0][j]='nan'
    
        
    asa = np.array(a)   #OMEGAa=a.*OMEGA
    OMEGAa=asa*ww
    OMEGAab=np.reshape(OMEGAa,(1,np.prod(OMEGAa.shape)))  #retains NaN
    avarres[0][i]=np.nanmean(OMEGAab) #

    asa = np.array(a)  
    OMEGAat=asa*wt
    OMEGAatb=np.reshape(OMEGAat,(1,np.prod(OMEGAat.shape)))  #retains NaN
    avarrest[0][i]=np.nanmean(OMEGAatb) 

x = range(0,pers,1)
xa =np.array(x)
#fig1= plt.plot(xa, avarres.transpose())
plt.plot(xa, avarres.transpose(),'b') 
plt.plot(xa, avarrest.transpose(),'r')

plt.xlabel('Lag');
plt.ylabel('Average auto-covariance function');

plt.show()
   
D = OMEGA10.shape
D0 = list(D)
D1 = D0[0] 
D2 = D0[1]
DM= np.ones((D1,D2))
UDM = np.triu(DM)
xw= np.array(OMEGA10[np.triu_indices(C1)])[np.newaxis]/NumberOfHouseholds
tavarres=np.zeros((1, pers))

#FIGURE: Autocovariancefunction by lag
for i in range(0,pers-1,1):
    q[0][:] = lag[0][:]
    n1 = q.shape
    n2 = list(n1) 
    n4 = n2[1]    
    for j in range(0,n4,1):   
        if q[0][j]==i:
           q[0][j]=1 
        else:
           q[0][j]='nan'
    #OMEGAa=a.*OMEGA  
    qsq = np.array(q)
    OMEGAac=qsq*xw
    OMEGAad=np.reshape(OMEGAac,(1,np.prod(OMEGAac.shape)))  #retains NaN
    #ndat = np.ma.masked_array(OMEGAa,np.isnan(OMEGAa))
    tavarres[0][i]=np.nanmean(OMEGAad) #burasi sorun yaratabilir    

        
    #FIGURE: Autocovariancefunction by lag
xx = range (0,pers,1)
xxa = np.array(xx)
    # Plot y1 vs. x (blue, solid) and y2 vs. x (red, dashed)

fig2 = plt.plot(xxa, tavarres.transpose())
    #Add title and axis labels
    #title('Trigonometric Functions', 'fontsize', 10);
plt.xlabel('Lag')
plt.ylabel('Average auto-covariance function Lag 5')


#END FIGURE
#end of autocovariance start of estimation matrices

moms=xw.transpose() #you should add () at the end or not verify
SPAN=pers*(pers+1)/2
start=np.zeros((nparms,1))

start[0][0]=0  #AR(1)	
start[1][0]=0  #MA(1)                                           
start[2][0]=0                                  

start[2:NumberOfPeriods+2,0]=0.06 #*np.ones((NumberOfPeriods,1))   
start[NumberOfPeriods+3:,0]=0.04 #*np.ones((NumberOfPeriods-2,1))   #write until the end
	

momd=moms                               #SPACE for the DATA MOMENTS
W=np.identity(SPAN)                     #IDENTITY WEIGHT MATRIX


lb = np.zeros((1,nparms))
lb[0][0] = -0.99
lb[0][1] = -0.99
lb[0][2] = 0
lb = lb.transpose()
        
        
ub = 0.2*np.ones((1,nparms))
ub[0][0] = 0.99
ub[0][1] = 0.99
ub[0][2] = 0.1
ub = ub.transpose()  
## - end of autocovariance

##-start of shock estimation
d= 4 #the lag of our choice in the data it can be at most 4
deltaALL4 = np.zeros((NumberOfPeriods-4, NumberOfHouseholds)) #lag of 4
deltaALL3 = np.zeros((NumberOfPeriods-3, NumberOfHouseholds)) #lag of 3
deltaALL2 = np.zeros((NumberOfPeriods-2, NumberOfHouseholds)) #lag of 2
deltaALL1 = np.zeros((NumberOfPeriods-1, NumberOfHouseholds)) #lag of 1

#trans storeage
deltaTRA4 = np.zeros((NumberOfPeriods-4, NumberOfHouseholds)) #lag of 4
deltaTRA3 = np.zeros((NumberOfPeriods-3, NumberOfHouseholds)) #lag of 3
deltaTRA2 = np.zeros((NumberOfPeriods-2, NumberOfHouseholds)) #lag of 2
deltaTRA1 = np.zeros((NumberOfPeriods-1, NumberOfHouseholds)) #lag of 1

#perm storage
deltaPER4 = np.zeros((NumberOfPeriods-4, NumberOfHouseholds)) #lag of 4
deltaPER3 = np.zeros((NumberOfPeriods-3, NumberOfHouseholds)) #lag of 3
deltaPER2 = np.zeros((NumberOfPeriods-2, NumberOfHouseholds)) #lag of 2
deltaPER1 = np.zeros((NumberOfPeriods-1, NumberOfHouseholds)) #lag of 1

varianceALL = np.zeros((NumberOfHouseholds,4))
varianceTRA = np.zeros((NumberOfHouseholds,4))
variancePER = np.zeros((NumberOfHouseholds,4))

#the difference of shocks with respect to different lags
for i in range(0,NumberOfPeriods-4,1):    
        deltaALL4[i][:] =  ALLComponent[i+4][:] - ALLComponent[i][:]
        deltaTRA4[i][:] = TransitoryComponent[i+4][:] - TransitoryComponent[i][:]
        deltaPER4[i][:] = PermanentComponent[i+4][:] - PermanentComponent[i][:]
        
for i in range(0,NumberOfPeriods-3,1):    
        deltaALL3[i][:] =  ALLComponent[i+3][:] - ALLComponent[i][:]
        deltaTRA3[i][:] = TransitoryComponent[i+3][:] - TransitoryComponent[i][:]
        deltaPER3[i][:] = PermanentComponent[i+3][:] - PermanentComponent[i][:]
        
for i in range(0,NumberOfPeriods-2,1):    
        deltaALL2[i][:] =  ALLComponent[i+2][:] - ALLComponent[i][:]
        deltaTRA2[i][:] = TransitoryComponent[i+2][:] - TransitoryComponent[i][:]
        deltaPER2[i][:] = PermanentComponent[i+2][:] - PermanentComponent[i][:]
        
for i in range(0,NumberOfPeriods-1,1):    
        deltaALL1[i][:] =  ALLComponent[i+1][:] - ALLComponent[i][:]
        deltaTRA1[i][:] = TransitoryComponent[i+1][:] - TransitoryComponent[i][:]
        deltaPER1[i][:] = PermanentComponent[i+1][:] - PermanentComponent[i][:]

for i in range(1,NumberOfHouseholds):        
        varianceALL[i][0] = np.var(deltaALL1[:,i])
        varianceALL[i][1] = np.var(deltaALL2[:,i])
        varianceALL[i][2] = np.var(deltaALL3[:,i])
        varianceALL[i][3] = np.var(deltaALL4[:,i])
        
        varianceTRA[i][0] = np.var(deltaTRA1[:,i])
        varianceTRA[i][1] = np.var(deltaTRA2[:,i])
        varianceTRA[i][2] = np.var(deltaTRA3[:,i])
        varianceTRA[i][3] = np.var(deltaTRA4[:,i])
        
        variancePER[i][0] = np.var(deltaPER1[:,i])
        variancePER[i][1] = np.var(deltaPER2[:,i])
        variancePER[i][2] = np.var(deltaPER3[:,i])
        variancePER[i][3] = np.var(deltaPER4[:,i])
        
#delta matrices - rows are periods and columns are households
#variance matrices - rows are households and columns are lags starting from 1 to 4 
#for the estimator the sum of lags per every household will be taken 

#the variences are not the same so that is why I cannot use the simplified formula

#now to estimate the income risk - that is var(delta U)

#THIS PART CONSIDERS BOTH THE TRANSITORY SHOCK AND THE PERMENANT SHOCK 
vardeltau1 = np.zeros((NumberOfHouseholds,1)) #for every household there will be one variance
vardeltau2 = np.zeros((NumberOfHouseholds,1)) # with different lag lengths from 1 to 4 
vardeltau3 = np.zeros((NumberOfHouseholds,1))
vardeltau4 = np.zeros((NumberOfHouseholds,1))

#here I can manipulate to have trans and perm shocks (or only one)
vardeltau1 = variancePER[:,0] + varianceTRA[:,0]*2 #lag 1
vardeltau2 = variancePER[:,0] + variancePER[:,1] + varianceTRA[:,0] + varianceTRA[:,1] #lag2
vardeltau3 = variancePER[:,0] + variancePER[:,1] + variancePER[:,2] + varianceTRA[:,0] + varianceTRA[:,2] #lag3
vardeltau4 = variancePER[:,0] + variancePER[:,1] + variancePER[:,2] + variancePER[:,3] + varianceTRA[:,0] + varianceTRA[:,3] #lag4

#estimation and robustness purposes
#decomposing the variances into PER and TRA to show the estimates are correct. 
#with the random data generation 

#persistent estimation - initialization
pvardeltau1 = np.zeros((NumberOfHouseholds,1)) #for every household there will be one variance
pvardeltau2 = np.zeros((NumberOfHouseholds,1)) # with different lag lengths from 1 to 4 
pvardeltau3 = np.zeros((NumberOfHouseholds,1))
pvardeltau4 = np.zeros((NumberOfHouseholds,1))

#transitory estimation - initialization
tvardeltau1 = np.zeros((NumberOfHouseholds,1)) #for every household there will be one variance
tvardeltau2 = np.zeros((NumberOfHouseholds,1)) # with different lag lengths from 1 to 4 
tvardeltau3 = np.zeros((NumberOfHouseholds,1))
tvardeltau4 = np.zeros((NumberOfHouseholds,1))

#PER
pvardeltau1 = variancePER[:,0] #lag 1
pvardeltau2 = variancePER[:,0] + variancePER[:,1]#lag2
pvardeltau3 = variancePER[:,0] + variancePER[:,1] + variancePER[:,2] #lag3
pvardeltau4 = variancePER[:,0] + variancePER[:,1] + variancePER[:,2] + variancePER[:,3]#lag4

#TRA
tvardeltau1 = varianceTRA[:,0]*2 #lag 1
tvardeltau2 = varianceTRA[:,0] + varianceTRA[:,1] #lag2
tvardeltau3 = varianceTRA[:,0] + varianceTRA[:,2] #lag3
tvardeltau4 = varianceTRA[:,0] + varianceTRA[:,3] #lag4

m1= np.mean(pvardeltau1)
m2= np.mean(pvardeltau2)
m3= np.mean(pvardeltau3)
m4= np.mean(pvardeltau4) #mean comparison between my results and the paper's

m5= np.mean(tvardeltau1)
m6= np.mean(tvardeltau2)
m7= np.mean(tvardeltau3)
m8= np.mean(tvardeltau4) #mean comparison between my results and the paper's

#robustness check for the estimators (run 10 times)
print('')
print('')

v1 = np.std(pvardeltau1)
v2 = np.std(pvardeltau2)
v3 = np.std(pvardeltau3)
v4 = np.std(pvardeltau4)

v5 = np.std(tvardeltau1)
v6 = np.std(tvardeltau2)
v7 = np.std(tvardeltau3)
v8 = np.std(tvardeltau4)

print('')
print('')

data_matrix4 = [['Lag 1', m1, v1, m5, v5]]
#storage of mean and standard deviation for comparison on a table
t4 = Table(rows=data_matrix4, names=('Estimator', 'Mean PER', 'StdDev PER', 'Mean TRA' , 'StdDev TRA'))
print(t4)
print('')
print('')

#WELFARE EFFECTS
#now to test the change in transitory shock affects the welfare or not
#instead of persistent shock only I am going to input vardeltau or only the transitory shock 

#initializations
#taken from the paper's estimated values

mu = 0.005
betap = 0.99 #this is the beta as a parameter
gama1 = 1
gama2 = 2
tao = 0.1
deltasigma = 0.003 #the change of shock taken from the estimate 

#income risk here is taken as given for simplicity
#calculating the change in consumption to compansate the welfare loss
#equivalent change in lifetime consumption

#initialization 
deltac2lag1 = np.zeros((NumberOfHouseholds,1))
deltac2lag2 = np.zeros((NumberOfHouseholds,1))
deltac2lag3 = np.zeros((NumberOfHouseholds,1))
deltac2lag4 = np.zeros((NumberOfHouseholds,1))

deltac1lag1 = np.zeros((NumberOfHouseholds,1))
deltac1lag2 = np.zeros((NumberOfHouseholds,1))
deltac1lag3 = np.zeros((NumberOfHouseholds,1))
deltac1lag4 = np.zeros((NumberOfHouseholds,1))

#calculation
for i in range(1,NumberOfHouseholds): 
    #following calculations are for if gama is not equal to 1 #for different lags
    deltac2lag1[i][0] = (1-betap*((1+mu)**(1-gama2))*math.exp(0.5*gama2*(gama2-1)*(1+deltasigma)*vardeltau1[i])/(1-betap*((1+mu)**(1-gama2))*math.exp(0.5*gama2*(gama2-1)*vardeltau1[i])))**(1/1-gama2)-1
    deltac2lag2[i][0] = (1-betap*((1+mu)**(1-gama2))*math.exp(0.5*gama2*(gama2-1)*(1+deltasigma)*vardeltau2[i])/(1-betap*((1+mu)**(1-gama2))*math.exp(0.5*gama2*(gama2-1)*vardeltau2[i])))**(1/1-gama2)-1
    deltac2lag3[i][0] = (1-betap*((1+mu)**(1-gama2))*math.exp(0.5*gama2*(gama2-1)*(1+deltasigma)*vardeltau3[i])/(1-betap*((1+mu)**(1-gama2))*math.exp(0.5*gama2*(gama2-1)*vardeltau3[i])))**(1/1-gama2)-1
    deltac2lag4[i][0] = (1-betap*((1+mu)**(1-gama2))*math.exp(0.5*gama2*(gama2-1)*(1+deltasigma)*vardeltau4[i])/(1-betap*((1+mu)**(1-gama2))*math.exp(0.5*gama2*(gama2-1)*vardeltau4[i])))**(1/1-gama2)-1
    
    
    #following calculations are for if gamma is equal to 1 #for different lags
    deltac1lag1[i][0] = math.exp((betap/(1-betap)**2)*vardeltau1[i]*deltasigma/2)-1
    deltac1lag2[i][0] = math.exp((betap/(1-betap)**2)*vardeltau2[i]*deltasigma/2)-1
    deltac1lag3[i][0] = math.exp((betap/(1-betap)**2)*vardeltau3[i]*deltasigma/2)-1
    deltac1lag4[i][0] = math.exp((betap/(1-betap)**2)*vardeltau4[i]*deltasigma/2)-1


#THIS PART CONSIDERS ONLY PERMANENT SHOCKS
pvardeltau1 = np.zeros((NumberOfHouseholds,1)) #for every household there will be one variance
pvardeltau2 = np.zeros((NumberOfHouseholds,1)) # with different lag lengths from 1 to 4 
pvardeltau3 = np.zeros((NumberOfHouseholds,1))
pvardeltau4 = np.zeros((NumberOfHouseholds,1))

#here I can manipulate to have trans and perm shocks (or only one)
pvardeltau1 = variancePER[:,0]  #lag 1
pvardeltau2 = variancePER[:,0] + variancePER[:,1] #lag2
pvardeltau3 = variancePER[:,0] + variancePER[:,1] + variancePER[:,2] #lag3
pvardeltau4 = variancePER[:,0] + variancePER[:,1] + variancePER[:,2] + variancePER[:,3]#lag4


#initialization 
pdeltac2lag1 = np.zeros((NumberOfHouseholds,1))
pdeltac2lag2 = np.zeros((NumberOfHouseholds,1))
pdeltac2lag3 = np.zeros((NumberOfHouseholds,1))
pdeltac2lag4 = np.zeros((NumberOfHouseholds,1))

pdeltac1lag1 = np.zeros((NumberOfHouseholds,1))
pdeltac1lag2 = np.zeros((NumberOfHouseholds,1))
pdeltac1lag3 = np.zeros((NumberOfHouseholds,1))
pdeltac1lag4 = np.zeros((NumberOfHouseholds,1))

#calculation
for i in range(1,NumberOfHouseholds): 
    #following calculations are for if gama is not equal to 1 #for different lags
    pdeltac2lag1[i][0] = (1-betap*((1+mu)**(1-gama2))*math.exp(0.5*gama2*(gama2-1)*(1+deltasigma)*pvardeltau1[i])/(1-betap*((1+mu)**(1-gama2))*math.exp(0.5*gama2*(gama2-1)*pvardeltau1[i])))**(1/1-gama2)-1
    pdeltac2lag2[i][0] = (1-betap*((1+mu)**(1-gama2))*math.exp(0.5*gama2*(gama2-1)*(1+deltasigma)*pvardeltau2[i])/(1-betap*((1+mu)**(1-gama2))*math.exp(0.5*gama2*(gama2-1)*pvardeltau2[i])))**(1/1-gama2)-1
    pdeltac2lag3[i][0] = (1-betap*((1+mu)**(1-gama2))*math.exp(0.5*gama2*(gama2-1)*(1+deltasigma)*pvardeltau3[i])/(1-betap*((1+mu)**(1-gama2))*math.exp(0.5*gama2*(gama2-1)*pvardeltau3[i])))**(1/1-gama2)-1
    pdeltac2lag4[i][0] = (1-betap*((1+mu)**(1-gama2))*math.exp(0.5*gama2*(gama2-1)*(1+deltasigma)*pvardeltau4[i])/(1-betap*((1+mu)**(1-gama2))*math.exp(0.5*gama2*(gama2-1)*pvardeltau4[i])))**(1/1-gama2)-1
    
    
    #following calculations are for if gamma is equal to 1 #for different lags
    pdeltac1lag1[i][0] = math.exp((betap/(1-betap)**2)*pvardeltau1[i]*deltasigma/2)-1
    pdeltac1lag2[i][0] = math.exp((betap/(1-betap)**2)*pvardeltau2[i]*deltasigma/2)-1
    pdeltac1lag3[i][0] = math.exp((betap/(1-betap)**2)*pvardeltau3[i]*deltasigma/2)-1
    pdeltac1lag4[i][0] = math.exp((betap/(1-betap)**2)*pvardeltau4[i]*deltasigma/2)-1

#calculation of mean for comparison 
m2l1 = np.mean(deltac2lag1)
m2l2 = np.mean(deltac2lag2)
m2l3 = np.mean(deltac2lag3)
m2l4 = np.mean(deltac2lag4)

m1l1 = np.mean(deltac1lag1)
m1l2 = np.mean(deltac1lag2)
m1l3 = np.mean(deltac1lag3)
m1l4 = np.mean(deltac1lag4) #means for both shocks

pm2l1 = np.mean(pdeltac2lag1)
pm2l2 = np.mean(pdeltac2lag2)
pm2l3 = np.mean(pdeltac2lag3)
pm2l4 = np.mean(pdeltac2lag4)

pm1l1 = np.mean(pdeltac1lag1)
pm1l2 = np.mean(pdeltac1lag2)
pm1l3 = np.mean(pdeltac1lag3)
pm1l4 = np.mean(pdeltac1lag4) #means for permanent shocks only



#calculation of standard deviations for comparison 
s2l1 = np.std(deltac2lag1)
s2l2 = np.std(deltac2lag2)
s2l3 = np.std(deltac2lag3)
s2l4 = np.std(deltac2lag4)

s1l1 = np.std(deltac1lag1)
s1l2 = np.std(deltac1lag2)
s1l3 = np.std(deltac1lag3)
s1l4 = np.std(deltac1lag4) #std dev for both shocks

ps2l1 = np.std(pdeltac2lag1)
ps2l2 = np.std(pdeltac2lag2)
ps2l3 = np.std(pdeltac2lag3)
ps2l4 = np.std(pdeltac2lag4)

ps1l1 = np.std(pdeltac1lag1)
ps1l2 = np.std(pdeltac1lag2)
ps1l3 = np.std(pdeltac1lag3)
ps1l4 = np.std(pdeltac1lag4) #std dev for permanent shocks only

#storage and display 
data_matrix = [['Mean Lag 1', m1l1, pm1l1],
               ['Stddev Lag 1', s1l1, ps1l1]]
#storage of mean and standard deviation for comparison on a table
t = Table(rows=data_matrix, names=('Welfare Change with Gama = 1', 'TRA + PER', 'PER'))
print(t)
print('')
print('')

data_matrix2 = [['Mean Lag 1', m2l1, pm2l1],
               ['Stddev Lag 1', s2l1, ps2l1]]
#storage of mean and standard deviation for comparison on a table
t2 = Table(rows=data_matrix2, names=('Welfare Change with Gama = 2', 'TRA + PER', 'PER'))
print(t2)
print('')
print('')

#display of two effets on a graph 
xxc = range (0,NumberOfHouseholds,1)
xca = np.array(xxc)
 
#PER only is blue, #TRA + PER is red
plt.plot(xca,deltac1lag1,'r') 
plt.plot(xca,pdeltac1lag1,'b')
plt.xlabel('Number of Households')
plt.ylabel('Welfare Effect DeltaC with Gamma = 1, Lag 1')
plt.show()

plt.plot(xca,deltac1lag2,'r') 
plt.plot(xca,pdeltac1lag2,'b')
plt.xlabel('Number of Households')
plt.ylabel('Welfare Effect DeltaC with Gamma = 1, Lag 2')
plt.show()

plt.plot(xca,deltac1lag3,'r') 
plt.plot(xca,pdeltac1lag3,'b')
plt.xlabel('Number of Households')
plt.ylabel('Welfare Effect DeltaC with Gamma = 1, Lag 3')
plt.show()

plt.plot(xca,deltac1lag4,'r') 
plt.plot(xca,pdeltac1lag4,'b')
plt.xlabel('Number of Households')
plt.ylabel('Welfare Effect DeltaC with Gamma = 1, Lag 4')
plt.show()


#now for gamma = 2
#PER only is blue, #TRA + PER is red
plt.plot(xca,deltac2lag1,'r') 
plt.plot(xca,pdeltac2lag1,'b')
plt.xlabel('Number of Households')
plt.ylabel('Welfare Effect DeltaC with Gamma = 2, Lag 1')
plt.show()

plt.plot(xca,deltac2lag2,'r') 
plt.plot(xca,pdeltac2lag2,'b')
plt.xlabel('Number of Households')
plt.ylabel('Welfare Effect DeltaC with Gamma = 2, Lag 2')
plt.show()

plt.plot(xca,deltac2lag3,'r') 
plt.plot(xca,pdeltac2lag3,'b')
plt.xlabel('Number of Households')
plt.ylabel('Welfare Effect DeltaC with Gamma = 2, Lag 3')
plt.show()

plt.plot(xca,deltac2lag4,'r') 
plt.plot(xca,pdeltac2lag4,'b')
plt.xlabel('Number of Households')
plt.ylabel('Welfare Effect DeltaC with Gamma = 2, Lag 4')
plt.show()

