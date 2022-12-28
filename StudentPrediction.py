import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers



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

def create_model(lr, fl):
    model = tf.keras.models.Sequential()
    model.add(fl)
    # simple linear regressor
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))
    
    model.compile(optimizer=tf.keras.omptimizers.RMSprop(lr=lr),
                  lose="mean_squared_error",
                  metrics=[tf.keras.metrics.MeanSquaredError()])
    
    return model

def train_model(model, dataset, epochs, batch_size, label_name, validation_split):
    
    features = {name:np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features, y=label, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=validation_split)
    
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    
    return epochs, hist