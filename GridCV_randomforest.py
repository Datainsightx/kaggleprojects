#This code may be used to select the important features to keep
#Next step will be to use the selected features to build a prediction model

import numpy as np
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
#=============================================================================
#Load the data into DataFrames

train_users = pd.read_csv('C:/Users/Home/Desktop/train.csv')
test_users = pd.read_csv('C:/Users/Home/Desktop/test.csv')
#=============================================================================

df = pd.DataFrame (train_users)
y = df['target']

del df['ID']
del df['target']
X =df


#================================================================================
n_samples = 3000 #len(X)

# Build a classification task using 3 informative features
X, y = make_classification(n_samples,
                           n_features=131,
                           n_informative=3,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=True)

rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=100, oob_score = True) 

param_grid = { 
    'n_estimators': [200, 700],
    'max_features': ['auto', 'sqrt', 'log2']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X, y)

print (CV_rfc.best_params_)


# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=300,
                              random_state=0)


forest = forest.fit(X, y)

forest.feature_importances_ #array with importances of each feature

idx = np.arange(0, X.shape[1])
#create an index array, with the number of features

features_to_keep = idx[forest.feature_importances_ > np.mean(forest.feature_importances_)]
#only keep features whose importance is greater than the mean importance

x_feature_selected = X[:,features_to_keep]

print("n_samples used is..",n_samples)

print(features_to_keep)

model = SelectFromModel(forest, prefit=True)

train_new = model.transform(X)

poly = PolynomialFeatures()

train_poly = poly.fit_transform(train_new)

#train_poly = train_new
 
sorted_idx = np.argsort(forest.feature_importances_)
pos = np.arange(sorted_idx.shape[0]) 
plt.plot()
plt.bar(pos, forest.feature_importances_[sorted_idx], align='center')
plt.axis('tight')
plt.yticks(pos, df.columns.values[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Feature Importance')
#plt.show()


#===========================================================================================
#Build a model using transformed train data and selected feature from
#

extc = ExtraTreesClassifier(n_estimators=300,max_features= 70,criterion= 'entropy',min_samples_split= 5,
                            max_depth= 70, min_samples_leaf= 5)      

extc.fit(train_poly,y)

#rf = RandomForestClassifier(n_estimators=100)

#rf.fit(train_poly, y)

#calculate log loss for validation set

#print('Cross-validation logloss:', sklearn.metrics.log_loss(validlabels, y_pred))

#===========================================================================================
print("Done fitting Random Forest model, on to the next one")


df_test = pd.DataFrame (test_users)

ids = df_test['ID'].values

del df_test['ID']


columns = ['v28','v34','v69']
           #','v33','v37','v43','v49','v53','v56','v57',
           #'v58','v61','v63','v65','v67','v71','v82','v86','v90','v92','v93','v100','v103',
           #'v105','v108','v111','v114','v118','v121','v123','v130']

df = pd.DataFrame(df_test, columns=columns)


df2 = pd.get_dummies(df)

X_test = df2.apply(lambda x:x.fillna(x.value_counts().index[0]))

#X_test = X_test.fillna(-999)

poly = PolynomialFeatures()

test_poly = poly.fit_transform(X_test)

#============================================================================================= 

print('Predicting, how long have you got?')

y_pred = extc.predict_proba(test_poly)

#=============================================================================================

print('Saving the predictions to csv file...you have done well to wait')

pd.DataFrame({"ID": ids, "PredictedProb": y_pred[:,1]}).to_csv('GridCV_ETC.csv',index=False)

#y_predict = rf.predict_proba(test_poly)

#y_predict = rf.predict_proba(X_test)

print(y_pred[:,1])

print('Done now')




#Save predictions
#pd.DataFrame({"ID":ids, "PredictedProb": y_predict[:,1]}).to_csv('submission_GridCV.csv',index=False)




