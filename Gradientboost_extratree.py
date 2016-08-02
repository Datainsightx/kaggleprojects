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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import log_loss
#=============================================================================
#Load the data into DataFrames
print("Loading data...")

train_users = pd.read_csv('C:/Users/Home/Desktop/train.csv')
df = pd.DataFrame (train_users)
y = df['target']

del df['ID']
del df['target']

X = df

test_users = pd.read_csv('C:/Users/Home/Desktop/test.csv')
df_test = pd.DataFrame (test_users)
ids = df_test['ID'].values

del df_test['ID']
#=============================================================================

comb_data = pd.concat([X,df_test], axis=0)

exit()

#================================================================================
n_samples = 10000 #len(X)
n_estimators = 100

# Build a classification task using 3 informative features
X, y = make_classification(n_samples,shuffle=True)

rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=n_estimators, oob_score = True) 

param_grid = { 
    'n_estimators': [200, 700],
    'max_features': ['auto', 'sqrt', 'log2']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X, y)

print (CV_rfc.best_params_)


# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=n_estimators,random_state=0)

print("Make_classification done...")

forest = forest.fit(X, y)

forest.feature_importances_ #array with importances of each feature

idx = np.arange(0, X.shape[1])
#create an index array, with the number of features

features_to_keep = idx[forest.feature_importances_ > np.mean(forest.feature_importances_)]
#only keep features whose importance is greater than the mean importance

X = X[:,features_to_keep]


print(features_to_keep)

print("Features have now been selected...")
#===============================================================================
# One hot encoding of features selected

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train,
                                                            y_train,
                                                            test_size=0.5)

grd = GradientBoostingClassifier(n_estimators=n_estimators)
grd_enc = OneHotEncoder()
grd_lm = LogisticRegression()
grd.fit(X_train, y_train)
grd_enc.fit(grd.apply(X_train)[:, :, 0])
grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)

print("One hot encoding done...")

y_pred_grd_lm = grd_lm.predict_proba(
    grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)

print("Prediction done...")

plt.figure(1)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

score = log_loss(y_test, y_pred_grd_lm)

print("Logloss score is...",score)

exit()

#===============================================================================
#Applying model to actual test data




columns = ['v2','v5','v13','v15']
           

test_new = pd.DataFrame(df_test, columns=columns)

test_new = test_new.fillna(-999.9)
#=============================================================================================

print("One-hot encoding on test data being done...")

y_pred = grd_lm.predict_proba(
    grd_enc.transform(grd.apply(test_new)))

#=============================================================================================

print('Saving the predictions to csv file...you have done well to wait')

pd.DataFrame({"ID": ids, "PredictedProb": y_pred[:,1]}).to_csv('Boostingtrees.csv',index=False)


print(y_pred[:,1])

print("Prediction of test data completed")




