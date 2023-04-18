
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import cv2
from scipy import ndimage

dataset = pd.read_csv('C:/ML/RawDataCollectedFromSensors/UpperSensorDataPreprocessing/UpperSensorCombinedData.csv')




X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

PeakPositions = []
for j in range(3749):
    signal = X[j, :]
    
    window_length = 1024
    overlap = 0.5
    
    
    window = sig.windows.hamming(window_length)
    frequencies, times, spectrogram = sig.spectrogram(signal, fs=1, window=window, noverlap=int(window_length*overlap), nperseg=window_length, scaling='spectrum')
    
    # #plot the spectrogram
    # plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram), cmap='magma')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()
     
    #Finding the first echo of the Signal   
    echo_threshold = np.max(spectrogram) / 2
    echo_position = np.where(spectrogram > echo_threshold)
    echo_time_index = echo_position[1][0]
    echo_frequency_index = echo_position[0][0]
    echo_time = times[echo_time_index]
    echo_frequency = frequencies[echo_frequency_index]
    PeakPositions.append(echo_time)
    
    # #Scaling of the Spectrogram in Logarithmic
    # plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram), cmap='magma')
    # plt.plot([echo_time], [echo_frequency], 'r+', markersize=10)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()

FirstEcho = np.array(PeakPositions).T
FirstEcho_integer = FirstEcho.astype('int')
#drawing one ADC signal
signal = X[1, :]
time_values = range(len(signal))
plt.figure(figsize=(30,30)) 
plt.plot(time_values, signal)



#Generating Spectrogram only with the approximate width of the first echo
# for i in range(3749):
#     min_sp_range = FirstEcho_integer[i]-200
#     max_sp_range = FirstEcho_integer[i]+200
#     Signal = X[i, min_sp_range:max_sp_range]
    
#     # Define the length of the Hamming window and step size
#     window_len = 32

#     sliding_step = 16

  

#     # Create a Hamming window function
#     hamming_window = np.hamming(window_len)

#     # Apply the sliding Hamming window and FFT to the signal
#     psd_list = []
#     for j in range(0, len(Signal) - window_len, sliding_step):
#         windowed_signal = Signal[j:j+window_len] * hamming_window
#         fft_output = np.fft.fft(windowed_signal)
#         psd = np.square(np.abs(fft_output)) / window_len
#         psd_list.append(psd)
#     # Convert the PSD list to a matrix and plot the spectrogram
#     sp = np.array(psd_list).T
#     # resize the spectrogram
#     max_value = np.max(sp)
#     scaled_spectrogram = sp / max_value
#     resized_spectrogram = cv2.resize(scaled_spectrogram, (256, 256))
    
#     # Generating log spectrogram
#     log_spectrogram = np.log(resized_spectrogram + 1e-10)  # Add a small value to avoid taking the logarithm of zero
#     normalized_spectrogram = (log_spectrogram - np.min(log_spectrogram)) / np.ptp(log_spectrogram)
    
#     plt.imsave(f'C:/ML/RawDataCollectedFromSensors/UpperSensorDataPreprocessing/FirstEchoSpectrograms/spectrogram_{i}.png', normalized_spectrogram, cmap='magma')
#     np.save(f'C:/ML/RawDataCollectedFromSensors/UpperSensorDataPreprocessing/FirstEchoNumpySpectrograms/spectrogram_{i}.npy', normalized_spectrogram)
    
    
    
    
    












































# signal = X[1, :]
# # Load spectrogram image
# spectrogram = cv2.imread('C:/ML/RawDataCollectedFromSensors/LowerSensorDataPreprocessing/Spectrograms/spectrogram_0.png')



# # # Load the spectrogram
# spectrogram = np.load(f'C:/ML/RawDataCollectedFromSensors/LowerSensorDataPreprocessing/NumpySpectrograms/spectrogram_5.npy')


# # Normalize the spectrogram
# spectrogram = spectrogram / np.max(spectrogram)

# # Apply a threshold
# threshold = 0.5
# mask = spectrogram > threshold

# # Apply morphological operations
# from scipy import ndimage
# morph_kernel = np.ones((3, 3))
# mask = ndimage.binary_dilation(mask, structure=morph_kernel, iterations=2)
# mask = ndimage.binary_erosion(mask, structure=morph_kernel, iterations=1)

# # Find the position of the first echo
# echo_position = np.argmax(np.sum(mask, axis=0))

# # Calculate the sliding window index where the echo occurred
# window_size = 1024  # assuming a window size of 1024 samples
# hop_size = 512  # assuming a hop size of 256 samples
# sliding_window_index = int(np.ceil((echo_position + 1 - window_size) / hop_size))

# # Plot the results
# plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
# plt.xlabel('Sliding Window Index')
# plt.ylabel('Frequency (Hz)')
# plt.title('Spectrogram')
# plt.axvline(sliding_window_index, color='r')
# plt.text(sliding_window_index, 100, 'Echo', color='r')
# plt.show()

# print('The first echo occurred at sliding window index:', sliding_window_index)


















