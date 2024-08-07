import healpy as hp
import numpy as np
import copy
import matplotlib.pyplot as plt
from pylab import cm
import torch.nn as nn
from scipy.special import sph_harm, factorial
from scipy import ndimage
from scipy import signal
import time
import warnings
import s2wav
import torch
import jax.numpy as jnp
from s2wav.filters import filters_directional
from scipy.ndimage import zoom

def convolve_fields(field, nside, N, device, use_c_backend, L, J_min):  
    field = healpy_to_array(field, nside)

    use_c_backend = False
    m = field
    filter_ = filters_directional(L= nside*2,N= N,J_min= 3,lam = 2.0,spin = 0,spin0 = 0)
    L = nside*2
    # Compute wavelet coefficients
    if(use_c_backend):
        f_wav, f_scal = s2wav.analysis(jnp.array(m), L = L, N = N, nside = nside,sampling="healpix", filters = filter_)
        return f_wav
    f_wav, f_scal = s2wav.analysis(m, L = L, N = N, filters = filter_, J_min = J_min)
    return f_wav

def upsample_sphere_signal(signal, target_shape):
    """
    Upsamples a signal on the sphere to a target shape using bilinear interpolation.

    Args:
        signal (np.ndarray): Input signal with dimensions [n_phi, n_theta].
        target_shape (tuple): Target shape (m_phi, m_theta) where m_phi > n_phi and m_theta > n_theta.

    Returns:
        np.ndarray: Upsampled signal with dimensions [m_phi, m_theta].
    """
    n_phi, n_theta = signal.shape
    m_phi, m_theta = target_shape
    
    # Calculate zoom factors
    zoom_factor_phi = m_phi / n_phi
    zoom_factor_theta = m_theta / n_theta
    
    # Perform interpolation using bilinear method (order=1)
    upsampled_signal = zoom(signal, (zoom_factor_phi, zoom_factor_theta), order=1)
    
    return upsampled_signal

def healpy_to_array(healpix_map, nside):
    # Number of pixels in the HEALPix map
    npix = hp.nside2npix(nside)
    
    # Determine the number of theta and phi bins
    n_theta = 2 * nside
    n_phi = 4 * nside
    
    # Create arrays for theta and phi
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    
    # Initialize the output array
    array = np.zeros((n_theta, n_phi))
    
    # Convert theta and phi to indices
    theta_idx = np.floor(theta / np.pi * n_theta).astype(int)
    phi_idx = np.floor(phi / (2 * np.pi) * n_phi).astype(int)
    
    # Fill the output array with the healpix_map values
    array[theta_idx, phi_idx] = healpix_map
    
    return array

def convert_to_pix_coord(ra, dec, nside=512):
    """
    Converts RA, DEC to HEALPix pixel coordinates.

    Args:
        ra (array): Right Ascension values.
        dec (array): Declination values.
        nside (int, optional): HEALPix nside parameter. Defaults to 512.

    Returns:
        array: HEALPix pixel coordinates.
    """
    theta = (90.0 - dec) * np.pi / 180.
    phi = ra * np.pi / 180.
    pix = hp.ang2pix(nside, theta, phi, nest=False)
    return pix

def IndexToDeclRa(index, nside, nest=False):
    """
    Converts HEALPix index to Declination and Right Ascension.

    Args:
        index (array): HEALPix pixel indices.
        nside (int): HEALPix nside parameter.
        nest (bool, optional): Nesting scheme of the HEALPix pixels. Defaults to False.

    Returns:
        tuple: Declination and Right Ascension.
    """
    theta, phi = hp.pixelfunc.pix2ang(nside, index, nest=nest)
    return theta, phi
    #return -np.degrees(theta - np.pi / 2.), np.degrees(phi)

def healpix_to_tensor(in_tensor, nside, n_bins):
    nest = False
    index = np.arange(hp.nside2npix(nside))
    polar = np.linspace(0,2*np.pi,n_bins)
    azimuthal = np.linspace(0,np.pi,n_bins)
    polar_array = []
    azimuthal_array = []
    for p in polar:
        for a in azimuthal:
            polar_array.append(p)
            azimuthal_array.append(a)

    azimuthal_array = np.array(azimuthal_array)
    polar_array = np.array(polar_array)
    index = hp.pixelfunc.angle2pix(nside,polar_array,azimuthal_array)
    values_of_the_map = in_tensor[index]

    values_of_the_map = values_of_the_map.reshape(n_bins,n_bins)
    return values_of_the_map

class WaveletPhaseHarmonics:
  """
  WaveletPhaseHarmonics is a class performing the calculations for the 5
  moments of the wavelet phase harmonics statistical method on a sphere
  """
  def __init__(self,tensor_field, J, N, J_min, nside, device = 'cpu', use_c_backend = False, L = 0):
    #Parameters passed in through constructor
      self.tensor_field = tensor_field
      self.J = J
      self.N = N
      self.J_min = J_min
      self.nside = nside
      self.L = L
      self.device = device
      self.Moments = dict()
      self.Indices = dict()
      #Taus
      self.tau_s_0_0 = self.fill_tau([2])
      self.tau_s_0_1 = self.fill_tau([0])
      self.tau_s_1_1 = self.fill_tau([2])
      self.tau_c_0_0 = self.fill_tau([2])
      self.tau_c_0_1 = self.fill_tau([2])
      #This is the field of convolutions, rather than an actual map
      self.use_c_backend = use_c_backend
      self.field = convolve_fields(field=self.tensor_field, nside=self.nside, N=self.N, device = self.device, use_c_backend = self.use_c_backend, L = self.L, J_min = self.J_min)
      
  
  def fill_tau(self, j_list = [0]):
        '''
        In various papers, the value of tau is fixed for a given moment or the
        authors choose to specify that for any given j-scale, the value of tau
        ranges by a certain amount.
        Inputs: j_list
        If the length of j_list is 1, this value corresponds to the value of
        delta_n for all scales
        If the length of j_list is J, this function
        Returns: 3D array storing each combination of j, delta_n, alpha
        '''

        alphas = [-np.pi/4, 0, np.pi/4, np.pi/2]
        #for each j
        if len(j_list) == 1:
            tau = list()
            if j_list[0] == 0:
                temp = numpy.empty(shape = (1,1), dtype = tuple)
                temp[0][0] = (0,0)
                tau.append(temp)
                #tau = numpy.empty(shape = (1,1,1), dtype = tuple)
                #tau[0][0][0] = (0,0)
                return tau

            tau = list()
            delta_n = j_list[0]
            temp = numpy.empty(shape = (delta_n, 4),  dtype = tuple)
            #tau = numpy.empty(shape = (1, delta_n, 4), dtype = tuple)
            for n in range(delta_n):
                for i in range(4):
                    temp[n][i] = (n, alphas[i])

            tau.append(temp)

            return tau
        elif len(j_list) != J:
            # add a few lines to log base 2 the inputs so it can match formatting as Marco wanted
            raise Exception("Length of list must match number of scales")
        tau = list()
        #tau = numpy.empty(shape = (len(j_list), delta_n, 4), dtype = tuple)

        for j in j_list:
            if j == 0:
                temp = numpy.empty(shape = (1,1), dtype = tuple)
                temp[0][0] = 0
                tau.append(temp)
            else:
                temp = numpy.empty(shape = (j, 4), dtype = tuple)
                for n in range(j):
                    for i in range(4):
                        temp[n][i] = (n, alphas[i])
                #tau[j] = temp[n][i]
                tau.append(temp)
        return tau
  
  def calculate_moments(self, J = -1,L = -1):
      if(J != -1):
        self.J = J
      if(L != -1):
        self.L = L
      J = self.J
      L = self.L
      (s_0_0_moments, s_0_0_indices) = self.S_0_0_moments_calculator(self.tau_s_0_0, self.J, self.L)
      (s_0_1_moments, s_0_1_indices) = self.S_0_1_moments_calculator(self.tau_s_0_1, self.J, self.L)
      (s_1_1_moments, s_1_1_indices) = self.S_1_1_moments_calculator(self.tau_s_1_1, self.J, self.L)
      (c_0_1_moments, c_0_1_indices) = self.C_0_1_moments_calculator(self.tau_s_0_1, self.tau_c_0_1, self.J, self.L)
      (c_0_0_moments, c_0_0_indices) = self.C_0_0_moments_calculator(self.tau_c_0_0, self.J, self.L)
      
      self.Moments['S00'] = s_0_0_moments
      self.Moments['S01'] = s_0_1_moments
      self.Moments['S11'] = s_1_1_moments
      self.Moments['C01'] = c_0_1_moments
      self.Moments['C00'] = c_0_0_moments

      self.Indices['S00'] = s_0_0_indices
      self.Indices['S01'] = s_0_1_indices
      self.Indices['S11'] = s_1_1_indices
      self.Indices['C01'] = c_0_1_indices
      self.Indices['C00'] = c_0_0_indices


  def calculate_s00(self, J = -1, L = -1):
      if(J != -1):
        self.J = J
      if(L != -1):
        self.L = L
      J = self.J
      L = self.L
      (s_0_0_moments, s_0_0_indices) = self.S_0_0_moments_calculator(self.tau_s_0_0, self.J, self.L)
      self.Moments['S00'] = s_0_0_moments
      self.Indices['S00'] = s_0_0_indices

  def calculate_s01(self, J = -1, L = -1):
      if(J != -1):
        self.J = J
      if(L != -1):
        self.L = L
      J = self.J
      L = self.L
      (s_0_1_moments, s_0_1_indices) = self.S_0_1_moments_calculator(self.tau_s_0_1, self.J, self.L)
      self.Moments['S01'] = s_0_1_moments
      self.Indices['S01'] = s_0_1_indices

  def calculate_s11(self, J = -1, L = -1):
      if(J != -1):
        self.J = J
      if(L != -1):
        self.L = L
      J = self.J
      L = self.L
      (s_1_1_moments, s_1_1_indices) = self.S_1_1_moments_calculator(self.tau_s_1_1, self.J, self.L)
      self.Moments['S11'] = s_1_1_moments
      self.Indices['S11'] = s_1_1_indices
  
  def calculate_c01(self, J = -1, L = -1):
      if(J != -1):
        self.J = J
      if(L != -1):
        self.L = L
      J = self.J
      L = self.L
      (c_0_1_moments, c_0_1_indices) = self.C_0_1_moments_calculator(self.tau_s_0_1, self.tau_c_0_1, self.J, self.L)
      self.Moments['C01'] = c_0_1_moments
      self.Indices['C01'] = c_0_1_indices

  def calculate_c00(self, J = -1, L = -1):
      if(J != -1):
        self.J = J
      if(L != -1):
        self.L = L
      J = self.J
      L = self.L
      (c_0_0_moments, c_0_0_indices) = self.C_0_0_moments_calculator(self.tau_c_0_0, self.J, self.L)
      self.Moments['C00'] = c_0_0_moments
      self.Indices['C00'] = c_0_0_indices

  def get_coeffs(self, st):
    x = self.Moments.get(st)
    return (self.Moments.get(st), self.Indices.get(st))


  def get_all_coeffs(self):
    s00, s00_indices = self.get_coeffs("S00")
    s01, s01_indices = self.get_coeffs("S01")
    s11, s11_indices = self.get_coeffs("S11")
    c01, c01_indices = self.get_coeffs("C01")
    c00, c00_indices = self.get_coeffs("C00")
    coeffs = s00 + s01 + s11 + c01 + c00
    return coeffs

  def S_0_0_moments_calculator(self, tau_s_0_0, J, L):
    #field = convolve_fields(field=self.tensor_field, nside=self.nside, N=self.L, device = self.device)
    field = self.field
    moments = list()
    j_list = list()
    for j in range(self.J_min, len(field)):
        for l in range(len(field[j])): 
            moment_1 = field[j][l]
            #print('Moment', moment_1)
            #print('j', j)
            moment_1 = abs(moment_1)
            moment_2 = abs(moment_1)
            moment = np.cov(moment_1, moment_2)
            moment = np.mean(moment)
            moments.append(moment)
            scales = (j,l)
            j_list.append(scales)
            
    return (moments, j_list)

  def S_0_1_moments_calculator(self, tau_s_0_1, J, L):
    field = self.field
    moments = list()
    j_list = list()
    for j in range(self.J_min, len(field)):
        for l in range(len(field[j])): 
            moment_1 = field[j][l]
            #print(moment_1)
            moment_2 = abs(moment_1)
            moment = np.cov(moment_1, moment_2)
            moment = np.mean(moment)
            moments.append(moment)
            scales = (j,l)
            j_list.append(scales)
    return (moments, j_list)
                         

  def S_1_1_moments_calculator(self, tau_s_1_1, J, L):
    field = self.field
    moments = list()
    j_list = list()
    for j in range(self.J_min, len(field)):
        for l in range(len(field[j])): 
            moment_1 = field[j][l]
            #print(moment_1)
            moment_2 = moment_1
            moment = np.cov(moment_1, moment_2)
            moment = np.mean(moment)
            moments.append(moment)
            scales = (j,l)
            j_list.append(scales)
    return (moments, j_list)

  def C_0_0_moments_calculator(self, tau_c_0_0, J, L):
    field = self.field
    moments = list()
    j_list = list()
    for j2 in range(self.J_min, len(field)):
        for j1 in range(self.J_min, j2):
            for l2 in range(len(field[j2])):
                for l1 in range(len(field[j2])):
                    if(l1 == l2):
                        len_1 = len(field[j2][l2])
                        len_2 = len(field[j2][l2][0])
                        moment_1 = field[j1][l1]
                        moment_1 = abs(moment_1)
                        moment_1 = upsample_sphere_signal(moment_1, (len_1, len_2))
                        moment_2 = field[j2][l2]
                        moment_2 = abs(moment_2)
                        moment = np.cov(moment_1, moment_2)
                        moment = np.mean(moment)
                        moments.append(moment)
                        j_list.append((j1, j2, l1, l2))
                    elif((l1 - l2) <= 2):
                        len_1 = len(field[j2][l2])
                        len_2 = len(field[j2][l2][0])
                        moment_1 = field[j1][l1]
                        moment_1 = abs(moment_1)
                        moment_1 = upsample_sphere_signal(moment_1, (len_1, len_2))
                        moment_2 = field[j2][l2]
                        moment_2 = abs(moment_2)
                        moment = np.cov(moment_1, moment_2)
                        moment = np.mean(moment)
                        moments.append(moment)
                        j_list.append((j1, j2, l1, l2))
    return (moments, j_list)
      
  def C_0_1_moments_calculator(self, tau_s_0_1, tau_c_0_1, J, L):
    field = self.field
    moments = list()
    j_list = list()
    for j2 in range(self.J_min, len(field)):
        for j1 in range(self.J_min, j2):
            for l2 in range(len(field[j2])):
                for l1 in range(len(field[j2])):
                    if(l1 == l2):
                        len_1 = len(field[j2][l2])
                        len_2 = len(field[j2][l2][0])
                        moment_1 = field[j1][l1]
                        moment_1 = abs(moment_1)
                        moment_1 = upsample_sphere_signal(moment_1, (len_1, len_2))
                        moment_2 = field[j2][l2]
                        moment = np.cov(moment_1, moment_2)
                        moment = np.mean(moment)
                        moments.append(moment)
                        j_list.append((j1, j2, l1, l2))
                    elif((l1 - l2) <= 2):
                        len_1 = len(field[j2][l2])
                        len_2 = len(field[j2][l2][0])
                        moment_1 = field[j1][l1]
                        moment_1 = abs(moment_1)
                        moment_1 = upsample_sphere_signal(moment_1, (len_1, len_2))
                        moment_2 = field[j2][l2]
                        moment = np.cov(moment_1, moment_2)
                        moment = np.mean(moment)
                        moments.append(moment)
                        j_list.append((j1, j2, l1, l2))
    return (moments, j_list)
print
