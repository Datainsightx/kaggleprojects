import random
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
rnd=57
maxCategories=5
train=pd.read_csv('C:/Users/Home/Desktop/train.csv')
test=pd.read_csv('C:/Users/Home/Desktop/test.csv')
random.seed(rnd)
train.index=train.ID
test.index=test.ID
del train['ID'], test['ID']
target=train.target
del train['target']

print('Preparing data...')

#prepare data
traindummies=pd.DataFrame()
testdummies=pd.DataFrame()

for elt in train.columns:
    vector=pd.concat([train[elt],test[elt]], axis=0)

    #count as categorial if number of unique values is less than maxCategories
    if len(vector.unique())<maxCategories:
        traindummies=pd.concat([traindummies, pd.get_dummies(train[elt],prefix=elt,dummy_na=True)], axis=1).astype('int8')
        testdummies=pd.concat([testdummies, pd.get_dummies(test[elt],prefix=elt,dummy_na=True)], axis=1).astype('int8')
        del train[elt], test[elt]
    else:
        typ=str(train[elt].dtype)[:3]
        if (typ=='flo') or (typ=='int'):
            minimum=vector.min()
            maximum=vector.max()
            train[elt]=train[elt].fillna(int(minimum)-2)
            test[elt]=test[elt].fillna(int(minimum)-2)
            minimum=int(minimum)-2
            traindummies[elt+'_na']=train[elt].apply(lambda x: 1 if x==minimum else 0)
            testdummies[elt+'_na']=test[elt].apply(lambda x: 1 if x==minimum else 0)
            

            #resize between 0 and 1 linearly ax+b
            a=1/(maximum-minimum)
            b=-a*minimum
            train[elt]=a*train[elt]+b
            test[elt]=a*test[elt]+b
        else:
            if (typ=='obj'):
                list2keep=vector.value_counts()[:maxCategories].index
                train[elt]=train[elt].apply(lambda x: x if x in list2keep else np.nan)
                test[elt]=test[elt].apply(lambda x: x if x in list2keep else np.nan)                
                traindummies=pd.concat([traindummies, pd.get_dummies(train[elt],prefix=elt,dummy_na=True)], axis=1).astype('int8')
                testdummies=pd.concat([testdummies, pd.get_dummies(test[elt],prefix=elt,dummy_na=True)], axis=1).astype('int8')
                
                #Replace categories by their weights
                tempTable=pd.concat([train[elt], target], axis=1)
                tempTable=tempTable.groupby(by=elt, axis=0).agg(['sum','count']).target
                tempTable['weight']=tempTable.apply(lambda x: .5+.5*x['sum']/x['count'] if (x['sum']>x['count']-x['sum']) else .5+.5*(x['sum']-x['count'])/x['count'], axis=1)
                tempTable.reset_index(inplace=True)
                train[elt+'weight']=pd.merge(train, tempTable, how='left', on=elt)['weight']
                test[elt+'weight']=pd.merge(test, tempTable, how='left', on=elt)['weight']
                train[elt+'weight']=train[elt+'weight'].fillna(.5)
                test[elt+'weight']=test[elt+'weight'].fillna(.5)
                del train[elt], test[elt]
            else:
                print('error', typ)
            
            
train2=pd.concat([train,traindummies], axis=1)
test_new=pd.concat([test,testdummies], axis=1)
del traindummies,testdummies

fed_data = pd.concat([train,test], axis=0)


y = target
X = train2



#==================================================================================
print("Starting make_classification...")

n_samples = 3000 #len(X)

n_estimators = 300

# Build a classification task using 3 informative features
X, y = make_classification(n_samples,shuffle=True)

rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=n_estimators, oob_score = True) 

param_grid = { 
    'n_estimators':[200, 700],
    'max_features':['auto', 'sqrt', 'log2']
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

X_new = X[:,features_to_keep]


print("Features have now been selected...",features_to_keep)
#===============================================================================
# Validation step

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.5)

X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train,
                                                            y_train,
                                                            test_size=0.5)

grd = GradientBoostingClassifier(n_estimators=n_estimators)

grd_lm = LogisticRegression()

grd.fit(X_train, y_train)

grd_lm.fit(X_train_lr, y_train_lr)

print("Cross validation...")

y_pred_grd_lm = grd_lm.predict_proba(X_test)

fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm[:,1])

print("Prediction done...")

score = log_loss(y_test, y_pred_grd_lm[:,1])

print("Logloss score is...",score)

plt.figure(1)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


#===============================================================================
#Applying model to actual test data

test_new = pd.DataFrame(test_new)

pred_test = test_new.iloc[:,[1,5,7,11]]

print("The shape of test data is...",pred_test)

y_pred = grd_lm.predict_proba(pred_test)

#=============================================================================================

print('Saving the predictions to csv file...you have done well to wait')

pd.DataFrame({"ID": ids, "PredictedProb": y_pred[:,1]}).to_csv('Boostingtrees.csv',index=False)


print(y_pred[:,1])

print("Prediction of test data completed")
