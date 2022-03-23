import pandas as pd
import numpy as np
import re

def getSuperHeroNameList():
    dcData  = pd.read_csv(
    "datasets/dc-wikia-data_csv.csv")
    marvelData = pd.read_csv(
        "datasets/marvel-wikia-data_csv.csv")
    # Concatenate all names to a list
    return marvelData['name'].tolist()+dcData['name'].tolist()
def processNames(names):
        
    distinctItems = set()
    for name in names:
        cleanedString = re.sub('[^a-zA-Z0-9 \n\.]', '', name)
        elements = cleanedString.split(' ')
        for element in elements:
            distinctItems.add(element.lower())

    print('Total distinct names: '+str(len(distinctItems)))
    return list(distinctItems)
def generateCharIndexDicts(nameElements):
    # Get all chars used
    chars = set(''.join(nameElements))
    chars = list(chars)
    chars += ['\n']
    # Create mapping dicts to use for quick lookups in the binary matrix generation
    indexToChar = {}
    for i in range(len(chars)):
        indexToChar[i] = chars[i]
    charToIndex = {v: k for k, v in indexToChar.items()}
    return indexToChar,charToIndex
def createBinaryMatrix(nameElements,MAX_NAME_LENGTH,indexToChar,charToIndex):
    # Create a zeros array with necessary shape as described by https://github.com/Frixoe/name-generator and other rnn examples

    binaryMatrix = np.zeros((len(nameElements), MAX_NAME_LENGTH, len(indexToChar.keys())))
    for rowIndex in range(binaryMatrix.shape[0]):
        hadError = False
        for colIndex in range(binaryMatrix.shape[1]):
            try:
                binaryMatrix[rowIndex,colIndex,charToIndex[nameElements[rowIndex][colIndex]]] = 1
            except:
                binaryMatrix[rowIndex, colIndex, charToIndex['\n']] = 1
                hadError = True
                break
        if not hadError:
            binaryMatrix[rowIndex, binaryMatrix.shape[1] - 1, charToIndex['\n']] = 1
    return binaryMatrix
def createXYTrainTest(binaryMatrix):
    # Generating training and testing data as described in the Canadian Name Generator with modifications
    # Reassigning the data to training samples X and targets Y
    x, y = np.zeros(binaryMatrix.shape), np.zeros(binaryMatrix.shape)
    x[:, :, :], y[:, :y.shape[1] - 1, :] = binaryMatrix[:, :, :], binaryMatrix[:, 1:, :]

    # Spliting so we use 80% of the data
    splitIndex = int(x.shape[0]//5)*4

    xTrain, yTrain, xTest, yTest = (x[:splitIndex, :, :], y[:splitIndex, :], x[splitIndex:, :, :], y[splitIndex:, :])
    return xTrain,yTrain,xTest,yTest