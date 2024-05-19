
# Wavelet Phase Harmonics Repository

This repository is used to perform calculations of Wavelet Phase Harmonic Moments for Data on the Sphere, using wavelets and convolution from S2WAV repository. 

# Wavlet Phase Harmonics 

Wavelet Phase Harmonics are calculations of the covariance between fields produced by different frequencies of wavelets convolved with an input field. It is a statistical method intended to draw out non-gaussian information, and in this repository we provide code to calculate the wavelet phase harmonic moments of fields that exist on the Sphere. 

Currently In Progress: Implementing Spatial Shift Input 
## Usage/Examples

Calculations are performed in [WaveletPhaseHarmonics](https://github.com/MichaelJacob914/WaveletPhaseHarmonics/blob/main/WaveletPhaseHarmonics.py)


Create a WaveletPhaseHarmonics object, and call the following functions to calculate the s00, s01, s11, c01, and c00 moments respectively. 

Parameters: 

`tensor_field` - Field to calculate moments for

`J` - Number of scales 

`L` - Number of orientations is given by 2L - 1

`J_min` - Minimum scale of wavelet to perform calculation with

`n_side` - n_side of tensor_field. NOTE: Increasing n_side causes significant increase to time taken per calculation

`device` - Specify 'JAX' to use JAX implementation of S2WAV on CPU, any other value will default to without JAX


``` python
wph = WaveletPhaseHarmonics(tensor_field = sim_init, J = 5, L = 2, J_min = 3, nside=nside, device = 'cpu')

wph.calculate_s00()
wph.calculate_s01()
wph.calculate_s11()
wph.calculate_c01()
wph.calculate_c00()
```

Alternatively, all the moments can be calculated in a single call with
``` python
wph.calculate_moments()

s00, s00_indices = wph.get_coeffs("S00")
s01, s01_indices = wph.get_coeffs("S01")
s11, s11_indices = wph.get_coeffs("S11")
c01, c01_indices = wph.get_coeffs("C01")
c00, c00_indices = wph.get_coeffs("C00")
```
and get methods for the moments and corresponding indices are provided above as well. 

Indices are of the above (j,l) where j is the scale of the wavelet used for the calculation and l is the angular orientation. For the C00 and C01 moments which use two wavelets, indices are of the form (j1, l1, j2, l2). 

Code for parallelization for multiplie simulations is provided in [fisher_S2Wav](https://github.com/MichaelJacob914/WaveletPhaseHarmonics/blob/main/run_S2Wav.py)

Code for generating Fisher Forecasts based on the coefficients is provided in [FisherForeCast](https://github.com/MichaelJacob914/WaveletPhaseHarmonics/blob/main/FisherForecast.py)

[Demo](https://github.com/MichaelJacob914/WaveletPhaseHarmonics/tree/main/demo) provides a sample field as well as code to perform a single calculation of the WaveletPhaseHarmonic moments 

## Acknowledgements

 - [s2wav](https://github.com/astro-informatics/s2wav)
 - MJ and SK thank Marco Gatti and Bhuvnesh Jain for their research support
 - map_512_test.npy is a test map courtesy of Marco Gatti and DES
 




 
