import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

featureColumns = []
learningRate = 0.01
epochs = 10
batchSize = 50
labelName = "G3"
validationSplit = 0.2

dataFrameM = pd.read_csv('studentData/student-mat.csv');
dataFrameM = pd.get_dummies(dataFrameM, columns=['Mjob', 'Fjob','reason','guardian']) # one hot encode given columns

dataFrameM = dataFrameM.sample(frac=1) # shuffle data

# print(dataFrameM)
print(dataFrameM.corr()['G3'][:-1])
# Data shows a 0.80 corr between G1 & G3
# Data shows a 0.90 corr between G2 & G3
# Other Correlations are < 0.5

nessesaryCols = ["G3","G1","G2"] # columns that are being used
dataFrameM = dataFrameM[nessesaryCols]
trainData, testData = train, test = train_test_split(dataFrameM, test_size=0.2)

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
                  metrics=[tf.keras.metrics.MeanSquaredError()])
    
    print("complete creating model")
    return model

def train_model(model, dataset, epochs, batchSize, labelName, validationSplit):
    
    features = {name:np.array(value) for name,value in dataset.items()}
    # print(features)
    label = np.array(features.pop(labelName))
    # features=tf.convert_to_tensor(features)
    model.fit(x=features, y=label, batch_size=batchSize, epochs=epochs, shuffle=True, validation_split=validationSplit)
    
    return 0


g1 = tf.feature_column.numeric_column("G1")
featureColumns.append(g1)

g2 = tf.feature_column.numeric_column("G2")
featureColumns.append(g2)

featureLayer = layers.DenseFeatures(featureColumns)

model = create_model(learningRate, featureLayer)
train_model(model, trainData, epochs, batchSize, labelName, validationSplit)