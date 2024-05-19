#General Import Statements
import glob
import os
#from Moments_analysis import moments_map
import numpy as np
import gc
import pickle
import healpy as hp
import matplotlib.pyplot as plt
import pylab as mplot
import getdist
from getdist.gaussian_mixtures import GaussianND
from getdist import plots, MCSamples
import copy
import torch


#Functions used to perform Fisher Analysis/test coefficients

#Function takes in data arrays for various moments and concatenates them together so that all coefficients for each
#field are in one array
def mergeCoeffs(s00 = None, s01 = None, s11 = None, c01 = None, c00 = None):

    lists = []
    if((s00 != None)): 
        lists.append(s00)
    if((s01 != None)): 
        lists.append(s01)
    if((s11 != None)): 
        lists.append(s11)
    if((c01 != None)): 
        lists.append(c01)
    if((c00 != None)): 
        lists.append(c00)
    temp_2 = []
    for index in range(len(s00)):
        coeffs = []
        for coeff_index in range(len(lists)):
            coeffs = coeffs + lists[coeff_index][index][0]
            
        temp = []
        temp.append(coeffs)
        temp_2.append(temp)
    return temp_2

#concatenates results into matrix to calculate covariance
def concatenate(matrix, list, multiplier = 1): 
    #multiplier is used since increasing it to ~10**6 yields non-zero determinant
    length = len(list[0][0])
    if multiplier == 1: 
        for i in range(1, len(list)):
            list_temp = list[i][0][start:num_coeffs]
            list_temp = [num * multiplier for num in list_temp]
            list_temp = torch.Tensor(list_temp).reshape((length,1))
            matrix = torch.cat((matrix, list_temp), 1)
    return matrix

#Use depending on formatting
def meanCalc(list):
    temp = []
    for i in list:
        mean = np.mean(i)
        temp.append(mean)
    return temp

#Takes slice of data vector to determine if there is linear dependence
def takeSlices(list, slice_start, slice_end):
    temp_2 = []
    for i in list:
        temp = []
        temp.append(i[0][slice_start:slice_end])
        temp_2.append(temp)
    return temp_2

#determine if there are duplicate coefficients within a given field
def CountFrequency(my_list, printer):
    # Creating an empty dictionary
    freq = {}
    list = []
    for i in range(len(my_list)):
        item = my_list[i]
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
            list.append(i)
    if(printer):
        for key, value in freq.items():
            if(value > 1): 
                print("% d : % d" % (key, value))
    return list

#determine if there are a specific coefficient is the same across fields
def compareFields(initial_list, index): 
    temp = []
    for i in range(len(initial_list)):
        temp.append(initial_list[i][0][index])
    return CountFrequency(temp, True)

def derivativeCalculator(list_1, list_2, difference):
    size = len(list_1[0][0])
    length = len(list_1)
    sum_1 = torch.Tensor(list_1[0][0]).reshape((size,1))
    sum_2 = torch.Tensor(list_2[0][0]).reshape((size,1))
    for i in range(1, len(list_1)):
        temp_1 = torch.Tensor(list_1[i][0]).reshape((size,1))
        temp_2 = torch.Tensor(list_2[i][0]).reshape((size,1))
        sum_1 = sum_1 + temp_1
        sum_2 = sum_2 + temp_2
    sum_1 = sum_1/length
    sum_2 = sum_2/length

    return (sum_1 - sum_2)/(difference)

def calculateCovariance(initial_list, start = 0, printer = False):
    #num_coeffs = Num of coefficients Desired
    num_coeffs = len(initial_list[0][0])
    length = num_coeffs - start

    #Creates matrix to concatenate data values, each row is a separate value with columns being observations
    matrix = torch.Tensor(initial_list[0][0][start:num_coeffs]).reshape((length,1))
    matrix = concatenate(matrix, initial_list)

    cov = np.cov(matrix, rowvar = True)
    if(printer):
        print('matrix', matrix.shape)
        print('cov shape', cov.shape)
        print('det', np.linalg.det(cov))
        
    return cov

#data path
path = ''
s00 = torch.load(path + 's00' + '_fiducial.csv')
s11 = torch.load(path + 's01' + '_fiducial.csv')
s01 = torch.load(path + 's11' + '_fiducial.csv')
c01 = torch.load(path + 'c01' + '_fiducial.csv')
c00 = torch.load(path + 'c00' + '_fiducial.csv')

coeffs = mergeCoeffs(s00 = s00, s01 = s01, s11 = s11, c01 = c01, c00 = c00)

calculateCovariance(coeffs)

#load derivative cellblock
path = 'Derivatives/nside64_J3to5_L2_'
s8_p_s00 = torch.load(path + 's00_s8p' + '.csv')
s8_m_s00 = torch.load(path + 's00_s8m' + '.csv')
om_p_s00 = torch.load(path + 's00_omp' + '.csv')
om_m_s00 = torch.load(path + 's00_omm' + '.csv')

s8_p_s01 = torch.load(path + 's01_s8p' + '.csv')
s8_m_s01 = torch.load(path + 's01_s8m' + '.csv')
om_p_s01 = torch.load(path + 's01_omp' + '.csv')
om_m_s01 = torch.load(path + 's01_omm' + '.csv')

s8_p_s11 = torch.load(path + 's11_s8p' + '.csv')
s8_m_s11 = torch.load(path + 's11_s8m' + '.csv')
om_p_s11 = torch.load(path + 's11_omp' + '.csv')
om_m_s11 = torch.load(path + 's11_omm' + '.csv')

s8_p_c01 = torch.load(path + 'c01_s8p' + '.csv')
s8_m_c01 = torch.load(path + 'c01_s8m' + '.csv')
om_p_c01 = torch.load(path + 'c01_omp' + '.csv')
om_m_c01 = torch.load(path + 'c01_omm' + '.csv')

s8_p_c00 = torch.load(path + 'c00_s8p' + '.csv')
s8_m_c00 = torch.load(path + 'c00_s8m' + '.csv')
om_p_c00 = torch.load(path + 'c00_omp' + '.csv')
om_m_c00 = torch.load(path + 'c00_omm' + '.csv')


# In[ ]:


list_s8_p = mergeCoeffs(s8_p_s00, s8_p_s01, s8_p_s11, s8_p_c01, s8_p_c00)
list_s8_m = mergeCoeffs(s8_m_s00, s8_m_s01, s8_m_s11, s8_m_c01, s8_m_c00)
list_om_p = mergeCoeffs(om_p_s00, om_p_s01, om_p_s11, om_p_c01, om_p_c00)
list_om_m = mergeCoeffs(om_m_s00, om_m_s01, om_m_s11, om_m_c01, om_m_c00)

fiducial = list_s8_p + list_s8_m + list_om_p + list_om_m
print(len(fiducial))

d_xi_ds8 = derivativeCalculator(list_s8_p, list_s8_m, 0.03)
d_xi_dom = derivativeCalculator(list_om_p, list_om_m, 0.02)


# In[ ]:


dv = np.mean(np.array(fiducial)[:,0,:].reshape(998,3,75),axis=1)
#print(dv.shape)
dv_s8_m= np.mean(np.array(list_s8_m)[:,0,:].reshape(200,3,75),axis=1)
dv_om_m= np.mean(np.array(list_om_m)[:,0,:].reshape(200,3,75),axis=1)
dv_s8_p= np.mean(np.array(list_s8_p)[:,0,:].reshape(200,3,75),axis=1)
dv_om_p= np.mean(np.array(list_om_p)[:,0,:].reshape(200,3,75),axis=1)

mplt.plot(np.mean(dv, axis = 0), label = 'fiducial')
mplt.plot(np.mean(dv_om_p, axis = 0), label = 'om_p')
mplt.plot(np.mean(dv_om_m, axis = 0), label = 'om_m')
mplt.plot(np.mean(dv_s8_p, axis = 0), label = 's8_p')
mplt.plot(np.mean(dv_s8_m, axis = 0), label = 's8_m')

#mplt.plot(list_s8_p[0][0], label = 's8_p_0')
#mplt.plot(list_s8_m[0][0], label = 's8_m_0')
#mplt.plot(list_om_m[0][0], label = 'om_m_0')
#mplt.plot(list_om_p[0][0], label = 'om_p_0')

mplt.legend()


# In[ ]:


def plotFisher(results):
    chains_ = []
    cases = results.keys()
    for case in results.keys():
        #imagine you have multiple cases (different DV with different choices of scales)
        mean = [0.3, 0.8]
        C_par = results[case]

        x = np.random.multivariate_normal(np.array(mean),C_par,30000)
        sig8_ = np.array(x[:,1])
        om_ = np.array(x[:,0])


        ssa = np.c_[om_.T,sig8_.T]
        samples_ = MCSamples(samples=ssa,weights=np.ones(30000), names = ['Om','sigma8'], labels = [r'\Omega_{\rm m}','\sigma_8'])

        chains_.append(samples_)     

    
    g = plots.getSubplotPlotter(width_inch=7)
    
    g.triangle_plot(chains_,['Om','sigma8'],filled=[False,False,False,False,False,False,True,True,True],legend_labels=cases,  contour_lws=[1.2,1.2,1.2,1.2,1.2,1.,1.],
            legend_loc='upper right',colors=['red','blue','black','black','black'],
            contour_ls =['-','-','-','-.','-'],contour_colors=['red','blue','black','black','black'],param_limits={'Om': [0.1,0.5], 
                    's8': [0.6,1.0]})


# In[ ]:


def calcFisherMatrix(d_xi_ds8, d_xi_dom, cov, name, penalty = False, num_sims = 1000):
    F = np.zeros((2,2))
    cov = cov

    if(penalty == True):
        num_coeffs = len(d_xi_ds8)
        pen = (num_sims - num_coeffs - 2)/(num_sims - 1)
        cov = cov/pen
        
    size = len(d_xi_ds8)
    # inverse measurement covariance
    P = np.linalg.inv(cov)
    # derivatives of the datavector
    #results = dict()
    #d_xi_ds8 = (results_2['s8p_noSC']-results_2['s8m_noSC'])/0.03
    #d_xi_dom = (results_2['Omp_noSC']-results_2['Omm_noSC'])/0.02
    
    temp_1 = np.matmul(P,d_xi_ds8)
    d_xi_ds8_T = d_xi_ds8.reshape((1,size))
    F[1,1] = np.matmul(d_xi_ds8_T,temp_1) 
    
    temp_2 = np.matmul(P,d_xi_dom)
    d_xi_dom_T = d_xi_dom.reshape((1,size))
    F[0,0] = np.matmul(d_xi_dom_T,temp_2) 

    .5 * (np.matmul(d_xi_ds8_T,temp_1)  + np.matmul(d_xi_dom_T,temp_2))
    F[0,1] = F[1,0] =  .5 * (np.matmul(d_xi_ds8_T,temp_1)  + np.matmul(d_xi_dom_T,temp_2))
    C_par_2 = np.linalg.inv(F)
    results[name] = C_par_2
    
    plotFisher(results)


# In[ ]:


#This cell is used to calculate Fisher Forecast Matrices with the addition of a new set of moments for each plot
#Currently uses derivatives as fiducial for calculations, modify fiducial accordingly
results = dict()
num_sims = 800
penalty = True
list_s8_p = mergeCoeffs(s8_p_s00)#, s8_p_s01, s8_p_s11, s8_p_c01, s8_p_c00)
list_s8_m = mergeCoeffs(s8_m_s00)#, s8_m_s01, s8_m_s11, s8_m_c01, s8_m_c00)
list_om_p = mergeCoeffs(om_p_s00)#, om_p_s01, om_p_s11, om_p_c01, om_p_c00)
list_om_m = mergeCoeffs(om_m_s00)#, om_m_s01, om_m_s11, om_m_c01, om_m_c00)

fiducial = list_s8_p + list_s8_m + list_om_p + list_om_m
cov = calculateCovariance(fiducial)
d_xi_ds8 = derivativeCalculator(list_s8_p, list_s8_m, 0.03)
d_xi_dom = derivativeCalculator(list_om_p, list_om_m, 0.02)
calcFisherMatrix(d_xi_ds8, d_xi_dom, cov, 's00', penalty, num_sims)


list_s8_p = mergeCoeffs(s8_p_s00, s8_p_s01)#, s8_p_s11, s8_p_c01, s8_p_c00)
list_s8_m = mergeCoeffs(s8_m_s00, s8_m_s01)#, s8_m_s11, s8_m_c01, s8_m_c00)
list_om_p = mergeCoeffs(om_p_s00, om_p_s01)#, om_p_s11, om_p_c01, om_p_c00)
list_om_m = mergeCoeffs(om_m_s00, om_m_s01)#, om_m_s11, om_m_c01, om_m_c00)

fiducial = list_s8_p + list_s8_m + list_om_p + list_om_m
cov = calculateCovariance(fiducial)
d_xi_ds8 = derivativeCalculator(list_s8_p, list_s8_m, 0.03)
d_xi_dom = derivativeCalculator(list_om_p, list_om_m, 0.02)
calcFisherMatrix(d_xi_ds8, d_xi_dom, cov, 's00 + s01', penalty, num_sims)

list_s8_p = mergeCoeffs(s8_p_s00, s8_p_s01, s8_p_s11)#, s8_p_c01, s8_p_c00)
list_s8_m = mergeCoeffs(s8_m_s00, s8_m_s01, s8_m_s11)#, s8_m_c01, s8_m_c00)
list_om_p = mergeCoeffs(om_p_s00, om_p_s01, om_p_s11)#, om_p_c01, om_p_c00)
list_om_m = mergeCoeffs(om_m_s00, om_m_s01, om_m_s11)#, om_m_c01, om_m_c00)

fiducial = list_s8_p + list_s8_m + list_om_p + list_om_m
cov = calculateCovariance(fiducial)
d_xi_ds8 = derivativeCalculator(list_s8_p, list_s8_m, 0.03)
d_xi_dom = derivativeCalculator(list_om_p, list_om_m, 0.02)
calcFisherMatrix(d_xi_ds8, d_xi_dom, cov, 's00, s01, s11', penalty, num_sims)

list_s8_p = mergeCoeffs(s8_p_s00, s8_p_s01, s8_p_s11, s8_p_c01)#, s8_p_c00)
list_s8_m = mergeCoeffs(s8_m_s00, s8_m_s01, s8_m_s11, s8_m_c01)#, s8_m_c00)
list_om_p = mergeCoeffs(om_p_s00, om_p_s01, om_p_s11, om_p_c01)#, om_p_c00)
list_om_m = mergeCoeffs(om_m_s00, om_m_s01, om_m_s11, om_m_c01)#, om_m_c00)

fiducial = list_s8_p + list_s8_m + list_om_p + list_om_m
cov = calculateCovariance(fiducial)
d_xi_ds8 = derivativeCalculator(list_s8_p, list_s8_m, 0.03)
d_xi_dom = derivativeCalculator(list_om_p, list_om_m, 0.02)
#calcFisherMatrix(d_xi_ds8, d_xi_dom, cov, 's00, s01, s11, c01')

list_s8_p = mergeCoeffs(s8_p_s00, s8_p_s01, s8_p_s11, s8_p_c01, s8_p_c00)
list_s8_m = mergeCoeffs(s8_m_s00, s8_m_s01, s8_m_s11, s8_m_c01, s8_m_c00)
list_om_p = mergeCoeffs(om_p_s00, om_p_s01, om_p_s11, om_p_c01, om_p_c00)
list_om_m = mergeCoeffs(om_m_s00, om_m_s01, om_m_s11, om_m_c01, om_m_c00)

fiducial = list_s8_p + list_s8_m + list_om_p + list_om_m
cov = calculateCovariance(fiducial)
d_xi_ds8 = derivativeCalculator(list_s8_p, list_s8_m, 0.03)
d_xi_dom = derivativeCalculator(list_om_p, list_om_m, 0.02)
#calcFisherMatrix(d_xi_ds8, d_xi_dom, cov, 'all')


# In[ ]:




