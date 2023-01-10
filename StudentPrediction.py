import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

featureColumns = []
learningRate = 0.005
epochs = 50
batchSize = 1
labelName = "G3"
validationSplit = 0.5

dataFrameM = pd.read_csv('studentData/student-mat.csv');
dataFrameM = pd.get_dummies(dataFrameM, columns=['Mjob', 'Fjob','reason','guardian']) # one hot encode given columns

dataFrameM = dataFrameM.sample(frac=1) # shuffle data

# print(dataFrameM)
print(dataFrameM.corr()['G2'][:-1])
# Data shows a 0.80 corr between G1 & G3
# Data shows a 0.90 corr between G2 & G3
# Other Correlations are < 0.5

nessesaryCols = ["G3","G2"] # columns that are being used
dataFrameM = dataFrameM[nessesaryCols]
trainData, testData =  train_test_split(dataFrameM, test_size=0.3)

"""
dataFrameP = pd.read_csv('studentData/student-por.csv');
print(dataFrameP)
print(dataFrameP.corr())
Data shows a 0.83 corr between G1 & G3
Data shows a 0.92 corr between G2 & G3
Other Correlations are < 0.5
"""

def create_model(learningRate, featureLayer):
    model = tf.keras.models.Sequential()
    model.add(featureLayer)
    # simple linear regressor
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))
    
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learningRate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    print("complete creating model")
    return model


def train_model(model, dataset, epochs, batchSize, labelName, validationSplit):
    
    features = {name:np.array(value) for name,value in dataset.items()}

    label = np.array(features.pop(labelName))
    
    model.fit(x=features, y=label, batch_size=batchSize, epochs=epochs, shuffle=True, validation_split=validationSplit)
    
    return 0


# Creating a feature layer | Using G2 as Parameters
featureColumns.append(tf.feature_column.numeric_column("G2"))
# Since G1 has a lower correlation than G2, data ends up trying to overfit even more to train w/ G1

featureLayer = layers.DenseFeatures(featureColumns)

model = create_model(learningRate, featureLayer)
train_model(model, trainData, epochs, batchSize, labelName, validationSplit)

# rmse and val_rmse both average around 2 with some variance
# difference in training comes from what is in Train Data vs in Validation Data as 
# some rows have a big difference in G2 & G3, while most have near same G2 & G3

testFeatures = {name:np.array(value) for name, value in testData.items()}
testLabel = np.array(testFeatures.pop(labelName))

print("\nTest Evaluation using Trained Model")
results = model.evaluate(testFeatures, testLabel, batch_size=batchSize)

# rmse from evaluation is averages around 2 with little variance

# Overall, the model does not do much and isn't very useful
# model just uses the value from G2 to guess G3 and the rmse in training, validating and evaluating
# changes depending on variance in data rather than how model is trained

# if I try using more & different paremeters, then model will try to overfit the training model
# as other parameters don't have a good correlation with G3