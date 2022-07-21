# ---------- IMPORTS ---------------------------------------------------------------------------------

import os
import math
import scipy
import argparse
import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import fft, fftfreq
from scipy import signal as sp

# ---------- FUNCTIONS TO COMPUTE BLOCKINESS ---------------------------------------------------------

def to_1d_signal(an_image:np.array, axis=0):
    """Function to flatten the difference of rows or columns of an image to a one dimensionnal signal

    Args:
        an_image (np.array): image in a numpy array. For the moment only one-channel images are supported
        axis (int, optional): 0 or 1 to flatten along rows or columns. Defaults to 0.
    """
    shape = an_image.shape
    if axis==0:
        for i in range(1, shape[0]):
            an_image[-i,:] = abs(an_image[-i,:] - an_image[-i-1,:])
        an_image[0,:] = 0
        signal_1d = an_image.flatten('C')

    elif axis==1:
        for j in range(1, shape[1]):
            an_image[:,-j] = abs(an_image[:,-j] - an_image[:,-j-1])
        an_image[:,0]
        signal_1d = an_image.flatten('F')

    return(signal_1d)


def make_segments(signal, N:int, n_0:int, step:int):
    """This function will create segments extracted from the input signal

    Args:
        signal (list or np.array): input array or list 
        N (int): length of the segments
        n_0 (int): offset
        step (int): step between two beginning segments

    Returns:
        list: list of list containing the segments
    """
    input_sequence = list(signal)
    end_point = n_0 + N
    segments = []
    i = 0

    while end_point < len(signal):
        segments.append(input_sequence[n_0+(i*step):n_0+(i*step)+N])
        end_point = n_0 + ((i+1)*step) + N
        i+=1

    return segments


def fast_fourier_transform(segment:list):
    """This function computes the FFT of a one-dimensionnal input signal

    Args:
        segment (list): segment on which we want to compute the FFT
    """
    return(fft(segment))


def power_spectrum(segment:list):
    """Computes the power spectrum of an input segment

    Args:
        segment (list): fft of an input segment

    Returns:
        list: power spectrum of the input having lenght of input length / 2 + 1
    """
    N = len(segment)
    segment_ps_all = list((np.abs(segment))**2)
    segment_ps = segment_ps_all[:(int(N/2)+1)]
    segment_ps[1:-1] = list(map(lambda x: 2*x, segment_ps[1:-1]))
    return segment_ps


def overall_estimated_ps(all_ps:list):
    """Function to average all the lists of power spectrums

    Args:
        all_ps (list): list of list of power spectrum
    """
    L = len(all_ps)
    overall_sum = [sum(ps) for ps in zip(*all_ps)]
    overall_average = list(map(lambda x: x/L, overall_sum))
    return(overall_average)


def median_smoothing(overall_power_spectrum:list, window_size=5):
    """Function to median filter a one-dimensionnal input signal

    Args:
        overall_power_spectrum (list): signal to median filter
    """
    filtered = sp.medfilt(overall_power_spectrum, kernel_size=window_size)
    return(filtered)


def blocking_measure(P:list, P_m:list):
    """Function to compute the overall blockiness measure based on the differences between
    the overall power spectrum and the smoothed overall power spectrum at the specific
    frequencies N/8, N/4, 3N/8 and N/2

    Args:
        P (list): overall power spectrum
        P_m (list): smoothed overall power spectrum

    Returns:
        float: measure of blockiness of the rows or columns
    """
    N = 2*len(P) - 2
    points = [N/8, N/4, 3*N/8, N/2]
    M = 0
    for i in points:
        i = int(i)
        M += 7/8*(P[i] - P_m[i])
    return M


def nearest_smaller_power_of_two(x):
    """This function computes the nearest inferior power of two for a given number

    Args:
        x (float): input number
    """
    powers = np.array([32*2**i for i in range(9)])
    if x in powers: return(x)
    res = np.argmax((powers/x)>=1)
    return(2**(res+4))

# ---------- MAIN ------------------------------------------------------------------------------------
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='path to folder with input images')
    args = parser.parse_args()

    image_path = args.image_path

    image = plt.imread(image_path)
    h, w, c = image.shape

    h_ = nearest_smaller_power_of_two(h)
    w_ = nearest_smaller_power_of_two(w)

    image = image[:h_, :w_, 0]

    segment_length = 256

    segments_rows = make_segments(to_1d_signal(image, axis=0), N=segment_length, n_0=0, step=4)
    segments_cols = make_segments(to_1d_signal(image, axis=1), N=segment_length, n_0=0, step=4)

    X_rows = [fast_fourier_transform(s) for s in segments_rows]
    X_cols = [fast_fourier_transform(s) for s in segments_cols]

    P_rows = overall_estimated_ps([power_spectrum(X_i) for X_i in X_rows])
    P_cols = overall_estimated_ps([power_spectrum(X_i) for X_i in X_cols])

    P_m_rows = median_smoothing(P_rows)
    P_m_cols = median_smoothing(P_cols)

    M_bv = blocking_measure(P_rows, P_m_rows)
    M_bh = blocking_measure(P_cols, P_m_cols)

    print(.5*M_bv + .5*M_bh)
