import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

LOADFILE = 'weights/RNN1.weights.h5'
TESTFILE = 'new_hand_landmarks_data.csv'        # FILE FOR TRAININGGGGG
SAVEFILE = 'weights/RNN1.weights.h5'



class Example:
    def __init__(self,data,output):
        self.data = data        # nested list
        self.output = toOutputList(output)
        self.length = len(data)


    def split(self):
        return data,output

    
def processData(FILE):
    """open FILE and process its data. """
    df = pd.read_csv(FILE)
    vidKeys = set(df[df.columns[0]])
    videos_raw = []
    dataset = []
    for key in vidKeys :
        raw_vid = df[(df[df.columns[0]] == key)]
        video_frames = []
        for i in range(len(raw_vid)):
            video_frames.append(list(raw_vid.iloc[i][1:-1]))
        dataset.append(Example(video_frames,int(list(raw_vid.iloc[0])[-1])))
        
    
    return dataset

def toOutputList(i):
    """turn i into a list where only the ith element is 1"""
    l = [0,0,0,0,0,0,0,0,0,0]
    l[i] = 1
    return l

def toInputArray(s) :
    """convert a string into an array"""
    arr = s.split()
    arr = [float(el) for el in arr]
    return arr


training = processData(TESTFILE)

X_train, y_train = np.array([ex.data for ex in training]),np.array([ex.output for ex in training])  # TRAINING DATA : X input ( 5,44)  Y output (10)


# Define RNN Model
model = keras.Sequential([
    keras.Input(shape = (5,42)),
    layers.SimpleRNN(64, activation="relu", return_sequences=True),
    layers.SimpleRNN(32, activation="relu"),  # Second RNN layer
    layers.Dense(10, activation="softmax", name = 'outputLayer')  # Output layer for classification
])

# Compile Model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
if LOADFILE != None:
    model.load_weights(LOADFILE)
# Print Model Summary
model.summary()

history = model.fit(X_train, y_train, epochs=250, batch_size=32 )#, validation_data=(X_test, y_test))

model.save_weights(SAVEFILE)
prediction = model.predict(np.expand_dims(X_train[0], axis=0)) # PUT PREDICTION DATA HEEEEERREEEE 
# Plot accuracy
plt.plot(history.history['accuracy'], label="Train Accuracy")
#plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
