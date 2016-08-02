import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
import scipy as sp
from sklearn import preprocessing
from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.metrics
#===========================================================================
#Load the data into DataFrames

train_ = pd.read_csv('C:/Users/Home/Desktop/train.csv')
df = pd.DataFrame (train_)
train = df.apply(pd.to_numeric, args=('coerce',))
y_train = train['target']

test_ = pd.read_csv('C:/Users/Home/Desktop/test.csv')
df_test = pd.DataFrame (test_)
test = df_test.apply(pd.to_numeric, args=('coerce',))
#==============================================================================

#Feature engineering steps to come in here


#==============================================================================
#Pre-processing of data prior to dimension reduction and PC selection
#==============================================================================

#Remove all categorical data
#train = train.select_dtypes(exclude=['object'])
#test = test.select_dtypes(exclude=['object'])

#Drop ID and target columns from training data and test data where applicable

X_train   = train.drop(['ID','target'],axis=1).values

ids = test['ID'].values

X_test    = test.drop(['ID'],axis=1).values

X_train = pd.DataFrame (X_train)
X_test = pd.DataFrame (X_test)

#Replace NaN with -1

train = X_train.fillna(-999.9)

test = X_test.fillna(-999.9)



X_trainstd = StandardScaler().fit_transform(train)

X_teststd = StandardScaler().fit_transform(test)


# PCA
sklearn_pca = RandomizedPCA(n_components=2)

sklearn_pca.fit(X_trainstd)

#5 PCs explain about 96% of the variation in the training data set

t_transf = sklearn_pca.fit_transform(X_trainstd)


#print(pca.explained_variance_)


test_transf = sklearn_pca.fit_transform(X_teststd)


#print("Variance explained by the selected no of principal components in test data is...")
#print(pca.explained_variance_ratio_) 

#X_pc=np.column_stack((X_test[:,0],X_test[:,1]))



#================================================================================                         
# Train a Random Forest model
#===============================================================================

from sklearn.ensemble import RandomForestClassifier


rf = RandomForestClassifier(n_estimators=100)

rf.fit(t_transf, y_train)

#calculate log loss for validation set

#print('Cross-validation logloss:', sklearn.metrics.log_loss(validlabels, y_pred))

#===========================================================================================
print("Done fitting Random Forest model, on to the next one")

print('Predicting targets...')

y_predict = rf.predict_proba(test_transf)

print(y_predict[:,1])

print('Saving the predictions to csv file...')

#Save predictions
pd.DataFrame({"ID":ids, "PredictedProb": y_predict[:,1]}).to_csv('submission_RF.csv',index=False)

#calculate log loss for test set

#print('Test logloss:', sklearn.metrics.log_loss(y_predict, validlabels))

exit()
#================================================================================                         
# Train a SVM model
#===============================================================================
print("Fitting SVM classifier to the training set...")

svc = SVC()

svc.fit(X_train, y_train)

print("Done fitting SVM classifier, on to the next one")

print('Predicting targets...')

y_predict = svc.predict(X_test)

print("The SVM predictions are here.....")
print(y_predict)

#==========================================================================
#Save predictions
pd.DataFrame({"ID": "ID", "PredictedProb": y_predict}).to_csv('submission_SVM.csv',index=False)

#=============================================================================
#Plot of PC1 versus PC2
         
plt.scatter(X_test[:,0],'ro', X_test[:,1],'bo')
plt.title('PCA analysis of PC1 and PC2 using test data')
plt.show()










