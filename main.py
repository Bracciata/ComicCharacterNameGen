import pandas as pd
import numpy as np
import re
import util
# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)
# import tensorflow as tf
# from tensorflow.keras import layers
names = util.getSuperHeroNameList()
print(names)
print('Total marked distinct characters: '+ str(len(names)))
nameElements = util.processNames(names)
# The max length is equal to the longest existing name plus five for now.
MAX_NAME_LENGTH = max(map(len, nameElements))-2

indexToChar, charToIndex = util.generateCharIndexDicts(nameElements)
# Pushing tokenized data in to embedded space
binaryMatrix = util.createBinaryMatrix(nameElements,MAX_NAME_LENGTH,indexToChar,charToIndex)

# Generating test train
xTrain, yTrain, xTest, yTest = util.createXYTrainTest(binaryMatrix)
