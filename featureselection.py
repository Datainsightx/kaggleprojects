#This code may be used to select the important features to keep
#Next step will be to use the selected features to build a prediction model

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

import pandas as pd
#Load the data into DataFrames
train_users = pd.read_csv('C:/Users/Home/Desktop/train.csv')
#test_users = pd.read_csv('C:/Users/Home/Desktop/test.csv')

df = pd.DataFrame (train_users)
y = df['target']
X = df

# Build a classification task using 3 informative features
X, y = make_classification(n_samples=1000,
                           n_features=20,
                           n_informative=2,
                           n_redundant=2,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=True)

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=10,
                              random_state=0)


forest.fit(X, y)

importance = forest.feature_importances_ #array with importances of each feature

plt.hist(importance)
plt.title("The importance of selected features")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()

idx = np.arange(0, X.shape[1])
#create an index array, with the number of features

features_to_keep = idx[importance > np.mean(importance)]
#only keep features whose importance is greater than the mean importance

x_feature_selected = X[:,features_to_keep]
#pull X values corresponding to the most important features
plt.hist(importance)
plt.title("The selected features")
plt.xlabel("Features")
plt.ylabel("Arbitrary")
plt.show()

