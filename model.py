import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np
import random
class RNNModelWithLSTM: # Model is based on the canadian name generator with variawble name changes to match our styling, parameter changes, and format changes
    def __init__(self,train,charToIndex,indexToChar,maxLength):
        self.maxLength = maxLength
        self.charToIndex= charToIndex
        self.indexToChar = indexToChar
        self.model = Sequential([
            Dense(100),
            Dense(50),
            LSTM(111, input_shape=(None, train.shape[2]), return_sequences=True),
            Dense(len(charToIndex), activation="softmax")
        ])
        self.model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])

    def predict(self,length):
        xPrediction = np.zeros((1, length, len(self.charToIndex)))
        prediction = [np.random.randint(len(self.charToIndex) - 1)]

        generated= [self.indexToChar[prediction[-1]]]
        for i in range(length):
            xPrediction[0, i, :][prediction[-1]] = 1
            predictioArray = self.model.predict(xPrediction[:, :, :], batch_size=1, verbose=0)
            predictioArray = predictioArray[0, i, :]
            p = np.argmax(predictioArray)
            prediction.append(p)
            generated.append(self.indexToChar[prediction[-1]])

        return "".join(generated)
    def train_model(self,xTrain,yTrain):
        # 5 being the number of epochs
        for epoch in range(10):
            print("Epoch "+str(epoch))
            # Fitting the model
            self.model.fit(x=xTrain, y=yTrain, batch_size=500, epochs=1, verbose=1)

            # Generate ten names of lengths from five chars to max length
            for i in range(10):
                length = random.randint(10,self.maxLength)
                print(f'Epoch {epoch} Name {i} of Length {length}')
                # Note output String length is less than length specified due to removal of new line characters
                print(self.predict(length).replace("\n",''))
        print("Training complete!")