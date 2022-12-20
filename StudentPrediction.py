import numpy as np
import pandas as pd

dataFrameM = pd.read_csv('studentData/student-mat.csv');
dataFrameP = pd.read_csv('studentData/student-por.csv');

print(dataFrameM.corr())
print(dataFrameP.corr())