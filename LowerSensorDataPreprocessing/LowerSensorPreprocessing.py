# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 18:17:07 2023

@author: user
"""
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import cv2
import os



dataset = pd.read_csv('LowerSensorCombinedData.csv')

Xl = dataset.iloc[:, :-1].values
yl = dataset.iloc[:, -1].values

#yplot = dataset.iloc[1597]
#xplot = np.arange(16369)
#plt.figure(figsize=(30,30)) 
#plt.plot(xplot, yplot)



#import ADC signal from csv file
for j in range(1):
    signal = Xl[j, :]
    # Define the length of the Hamming window and step size
    window_len = 1024

    sliding_step = 512

  

    # Create a Hamming window function
    hamming_window = np.hamming(window_len)

    # Apply the sliding Hamming window and FFT to the signal
    psd_list = []
    for i in range(0, len(signal) - window_len, sliding_step):
        windowed_signal = signal[i:i+window_len] * hamming_window
        fft_output = np.fft.fft(windowed_signal)
        psd = np.square(np.abs(fft_output)) / window_len
        psd_list.append(psd)
    
    # Convert the PSD list to a matrix and plot the spectrogram
    sp = np.array(psd_list).T
    # Taking first 512 data points (lower frequency) of PSD to generate spectrogram
    spectrogram = sp[:512]
    max_value = np.max(spectrogram)
    scaled_spectrogram = spectrogram / max_value
    resized_spectrogram = cv2.resize(scaled_spectrogram, (256, 256))
    # plt.imshow(resized_spectrogram,aspect='auto')
    # plt.xlabel('Time (window index)')
    # plt.ylabel('Frequency (Hz)')
    # plt.show()


    # Generating log spectrogram
    log_spectrogram = np.log(resized_spectrogram + 1e-10)  # Add a small value to avoid taking the logarithm of zero
    normalized_spectrogram = (log_spectrogram - np.min(log_spectrogram)) / np.ptp(log_spectrogram)
    
    # We are using  'magma' colormap
    plt.imshow(normalized_spectrogram, cmap='magma')
    plt.xlabel('Time (window index)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar()
    plt.show()
    # plt.imsave(f'C:/ML/RawDataCollectedFromSensors/LowerSensorDataPreprocessing/Spectrograms/spectrogram_{j}.png', normalized_spectrogram, cmap='magma')
    # np.save(f'C:/ML/RawDataCollectedFromSensors/LowerSensorDataPreprocessing/NumpySpectrograms/spectrogram_{j}.npy', normalized_spectrogram)
    
    
    








