import numpy as np
import util
import model
# Make numpy values easier to read according to online docs we found
np.set_printoptions(precision=3, suppress=True)

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
rnnModel = model.RNNModelWithLSTM(xTrain,charToIndex,indexToChar,MAX_NAME_LENGTH)
rnnModel.train_model(xTrain,yTrain)

# Evaluating the model
metrics = rnnModel.model.evaluate(x=xTest, y=yTest, batch_size=100)

print("RNN with LSTM Model metrics: ")
# This is directly from Canadian Name Generation
for m_name, m in zip(rnnModel.model.metrics_names, metrics):
    print("{}: {}".format(m_name, m))
