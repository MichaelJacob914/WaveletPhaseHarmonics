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

def convert(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

d = np.load('maps_fisher_64.npy', allow_pickle = True)

nside = 64


index = np.arange(hp.nside2npix(nside))
nest = False

list_full = list()

run_count = 0
number_of_runs = 200 #e.g. number of sims.
#Parallelization begins here

header = 'omm' #change this header to alter files read in
d_again = d[()]
F = d_again[header]

start_initial = time.time()
while run_count < number_of_runs:
    comm = MPI.COMM_WORLD
    if run_count+comm.rank<number_of_runs:
        start = time.time()
        sim_init = np.zeros(64**2*12)
        counter = 0
        for pixel in F[run_count][0]:
            sim_init[pixel] = F[run_count][1][counter]
            counter += 1
        #sim_init = F[run_count][1]
        #sim_init = hp.ud_grade(sim_init,nside_out=64)
        #sim = torch.from_numpy(sim_init).to(device).contiguous()
        wph = WaveletPhaseHarmonics(tensor_field = sim_init, J = 5, L = 2, J_min = 3, nside=nside, device = 'cpu')
        wph.calculate_s00()
        wph.calculate_s01()
        wph.calculate_s11()
        wph.calculate_c01()
        wph.calculate_c00()
        s00, s00_indices = wph.get_coeffs("S00")
        s01, s01_indices = wph.get_coeffs("S01")
        s11, s11_indices = wph.get_coeffs("S11")
        c01, c01_indices = wph.get_coeffs("C01")
        c00, c00_indices = wph.get_coeffs("C00")
        coeffs = list()
        coeffs.append(s00)
        coeffs.append(s01)
        coeffs.append(s11)
        coeffs.append(c01)
        coeffs.append(c00)
        list_full.append(coeffs)
        end = time.time()
        print("doing interation", run_count)
        print('time taken', end - start)
    run_count+=comm.size
    comm.bcast(run_count,root = 0)
    comm.Barrier() 

final = time.time()
total_time = convert(final - start_initial)
print('Total time for all runs: ', total_time)

#change file name
file_name = 'FILE_NAME.csv'
torch.save(list_full, file_name)

