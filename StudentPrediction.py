import numpy as np
import pandas as pd

dataFrameM = pd.read_csv('studentData/student-mat.csv');
dataFrameP = pd.read_csv('studentData/student-por.csv');

print(dataFrameM)
print(dataFrameM.corr())
# Data shows a 0.80 corr between G1 & G3
# Data shows a 0.90 corr between G2 & G3
# Other Correlations are < 0.5


print(dataFrameP)
print(dataFrameP.corr())
# Data shows a 0.83 corr between G1 & G3
# Data shows a 0.92 corr between G2 & G3
# Other Correlations are < 0.5