import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import cv2
from scipy import ndimage
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import confusion_matrix
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
    # plt.plot([echo_time], [echo_frequency], 'b+', markersize=10)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()

FirstEcho = np.array(PeakPositions).T
FirstEcho_integer = FirstEcho.astype('int')
# #drawing one ADC signal
# signal = X[2, :]
# time_values = range(len(signal))
# plt.figure(figsize=(30,30)) 
# plt.plot(time_values, signal)

spectrograms_list = []
for j in range(3749):
    sp = np.load(f'C:/ML/RawDataCollectedFromSensors/UpperSensorDataPreprocessing\FirstEchoNumpySpectrograms/spectrogram_{j}.npy')
    spectrograms_list.append(sp)
spectrograms = np.array(spectrograms_list)


# Split dataset into training, validation, and test sets
train_spectrograms, test_spectrograms, train_labels, test_labels = train_test_split(spectrograms, FirstEcho_integer, test_size=0.15, random_state=42)


#reshaping data for CNN input
train_spectrograms2 = train_spectrograms.reshape((train_spectrograms.shape[0], train_spectrograms.shape[1], train_spectrograms.shape[2], 1))
test_spectrograms2 = test_spectrograms.reshape((test_spectrograms.shape[0], test_spectrograms.shape[1], test_spectrograms.shape[2], 1))

#CNN Architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

#Training of the CNN
history = model.fit(train_spectrograms2, train_labels, batch_size=32, epochs=10,validation_split = 0.2)

#Testing of CNN
test_loss, test_acc = model.evaluate(test_spectrograms2, test_labels)
print('Test accuracy:', test_acc)


y_pred = model.predict(test_spectrograms2)



