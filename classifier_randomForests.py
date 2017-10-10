import sklearn as skl
import scipy as sc

# sklearn's random forest implementation
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import featureExtractor as fe

# number of languages in the dataset
numOutputClassesDefault = 5

clf = RandomForestClassifier(max_depth=16, random_state=0)
tX, tY, numFeatures = fe.getTrainableData()
clf.fit(tX, tY)

