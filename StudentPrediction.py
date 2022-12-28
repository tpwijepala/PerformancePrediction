import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers

feature_colums = []
learning_rate = 0.01
epochs = 10
batchSize = 50
labelName = "G3"
validationSpit = 0.2

dataFrameM = pd.read_csv('studentData/student-mat.csv');
dataFrameP = pd.read_csv('studentData/student-por.csv');

# print(dataFrameM)
# print(dataFrameM.corr())
# Data shows a 0.80 corr between G1 & G3
# Data shows a 0.90 corr between G2 & G3
# Other Correlations are < 0.5

data = pd.get_dummies(dataFrameM, columns=['Mjob', 'Fjob','reason','guardian'])
print(data)
print(data.corr()['G3'][:-1])

# print(dataFrameP)
# print(dataFrameP.corr())
# Data shows a 0.83 corr between G1 & G3
# Data shows a 0.92 corr between G2 & G3
# Other Correlations are < 0.5

def create_model(learningRate, featureLayer):
    model = tf.keras.models.Sequential()
    model.add(featureLayer)
    # simple linear regressor
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))
    
    model.compile(optimizer=tf.keras.omptimizers.RMSprop(lr=learningRate),
                  lose="mean_squared_error",
                  metrics=[tf.keras.metrics.MeanSquaredError()])
    
    return model

def train_model(model, dataset, epochs, batchSize, labelName, validationSplit):
    
    features = {name:np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(labelName))
    history = model.fit(x=features, y=label, batch_size=batchSize, epochs=epochs, shuffle=True, validation_split=validationSplit)
    
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    
    return epochs, hist