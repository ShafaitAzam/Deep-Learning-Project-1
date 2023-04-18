import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
dataset = pd.read_csv('UpperSensorCombinedData.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# Load spectrogram dataset
spectrograms_list = []
for j in range(0,3749):
    sp = np.load(f'C:/ML/RawDataCollectedFromSensors/UpperSensorDataPreprocessing/NumpySpectrograms/spectrogram_{j}.npy')
    spectrograms_list.append(sp)
spectrograms = np.array(spectrograms_list)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)


# Split dataset into training, validation, and test sets
train_spectrograms, test_spectrograms, train_labels, test_labels = train_test_split(spectrograms, encoded_Y, test_size=0.15, random_state=42)



train_spectrograms2 = train_spectrograms.reshape((train_spectrograms.shape[0], train_spectrograms.shape[1], train_spectrograms.shape[2], 1))
test_spectrograms2 = test_spectrograms.reshape((test_spectrograms.shape[0], test_spectrograms.shape[1], test_spectrograms.shape[2], 1))


#CNN Architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Training of the CNN
history = model.fit(train_spectrograms2, train_labels, batch_size=32, epochs=10,validation_split = 0.2)

# #Testing of CNN
# test_loss, test_acc = model.evaluate(test_spectrograms2, test_labels)
# print('Test accuracy:', test_acc)


y_pred = model.predict(test_spectrograms2)

Prediction_binary_array = np.where(y_pred >= 0.5, 1, 0)

cm = confusion_matrix(test_labels, Prediction_binary_array)
print(cm)

classes = np.unique(test_labels)

# plot the confusion matrix as a heatmap
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=classes, yticklabels=classes,
       ylabel='True label',
       xlabel='Predicted label')
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# add text labels to each cell
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2. else "black")
fig.tight_layout()
plt.show()



