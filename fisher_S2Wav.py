import healpy as hp
import numpy as np
import copy
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from pylab import cm
import torch.nn as nn
from scipy.special import sph_harm, factorial
from scipy import ndimage
from scipy import signal
import time
import numpy as np
import warnings
import torch
import WPH_S2Wav
from WPH_S2Wav import WaveletPhaseHarmonics
from mpi4py import MPI

d = np.load('maps_fisher_64.npy', allow_pickle = True)

nside = 64


index = np.arange(hp.nside2npix(nside))
nest = False

polar, azimuthal = hp.pixelfunc.pix2ang(nside, index, nest=nest)

#theta, phi = np.mgrid[0:np.pi:512j, 0:2 * np.pi:512j]

list_full = list()

run_count = 0
number_of_runs = 200 #e.g. number of sims.
#Parallelization begins here

header = 'omm' #change this header to alter files read in
d_again = d[()]
F = d_again[header]


while run_count < number_of_runs:
    comm = MPI.COMM_WORLD
    if run_count+comm.rank<number_of_runs:
        sim_init = np.zeros(64**2*12)
        counter = 0
        for pixel in F[run_count][0]:
            sim_init[pixel] = F[run_count][1][counter]
            counter += 1
        #sim_init = F[run_count][1]
        #sim_init = hp.ud_grade(sim_init,nside_out=64)
        #sim = torch.from_numpy(sim_init).to(device).contiguous()
        wph = WaveletPhaseHarmonics(tensor_field = sim_init, J = 5, L = 2, J_min = 3, azimuthal = azimuthal, polar = polar, nside=nside, device = 'cpu')
        #wph.s00_calculator()
        #wph.s01_calculator()
        #wph.s11_calculator()
        #wph.c01_calculator()
        wph.c00_calculator()
        #s00, s00_indices = wph.get_coeffs("S00")
        #s01, s01_indices = wph.get_coeffs("S01")
        #s11, s11_indices = wph.get_coeffs("S11")
        #c01, c01_indices = wph.get_coeffs("C01")
        c00, c00_indices = wph.get_coeffs("C00")
        list_temp = list()
        #list_temp.append(s00)
        #list_temp.append(s01)
        #list_temp.append(s11)
        #list_temp.append(c01)
        list_temp.append(c00)
        list_full.append(list_temp)
        print("doing interation", run_count)
    run_count+=comm.size
    comm.bcast(run_count,root = 0)
    comm.Barrier() 

#change file name
torch.save(list_full, 'nside64_J3to5_L2_c00_omm.csv')

