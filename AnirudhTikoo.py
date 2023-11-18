#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[2]:


#Libraries to be imported
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score, confusion_matrix,classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest,chi2,f_classif,mutual_info_classif
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,OneHotEncoder,Normalizer
from sklearn.naive_bayes import GaussianNB


# # Getting Data

# In[6]:


import pandas as pd
import numpy as np
data=pd.read_csv("C:/Users/AnmitY/OneDrive/Desktop/training_data.csv")
target=pd.read_csv("C:/Users/AnmitY/OneDrive/Desktop/training_data_targets.csv",names=['y'])


# # Preprocessing the data

# ## taking care of empty values

# In[7]:


median=data.median()
data=data.fillna(median)



# ## encoding the data

# In[8]:


gender=np.array(data.gendera).reshape(-1,1)


# In[9]:


#One hot encoder
ohd=OneHotEncoder()
gender=ohd.fit_transform(gender).toarray().astype(int)
gender=pd.DataFrame(gender)
add=ohd.fit_transform(data.iloc[:,3:12]).toarray().astype(int)
#dropping columns one hot encoded
data.drop(columns=data.columns[3:12],inplace=True)
add=pd.DataFrame(add)
#dropping gender
data.drop(columns=data.columns[1],inplace=True)
# # Feature Selection

data=SelectKBest(mutual_info_classif,k=34).fit_transform(data,target)


# ## Normalize the data
n=Normalizer(norm='l2',copy=True)
data=n.fit_transform(data)

data=pd.DataFrame(data)

#concatenating the arrays
data1=pd.concat([data,add,gender],axis=1)



# # PCA
pca=PCA(n_components=29)
data1=pca.fit_transform(data1)


# # Training

data1=np.array(data1)
target=np.array(target)
skf=StratifiedKFold(n_splits=4)
for train, test in skf.split(data1,target):
        x_train = data1[train]
        y_train = target[train]
        x_test = data1[test]
        y_test = target[test]


y_train=np.array(y_train)
y_train=y_train.reshape(-1)

x_train=pd.DataFrame(x_train)



# ## Getting Parameters for GridSearchCV


#Looking for Parameters

param_grid2={'n_neighbors':[5,6,7,10,12,15,16,17,18,20,23,24,34,56],'weights':['uniform', 'distance'],'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],'leaf_size':[5,6,7,8,9,10,11],'p':[1,2,3]}#knn
param_grid3={'kernel':['sigmoid','liner','poly', 'rbf'],'C':[0.1,1,5,10,15,20,30,40,100],'gamma':['scale', 'auto'],'coef0':[0,1,2,3],'shrinking':[True,False],'decision_function_shape':['ovo','ovr']}#SVC
param_grid4={'activation':['identity', 'logistic', 'tanh', 'relu'],'solver':[ 'sgd', 'adam'],'max_iter':[200,400,800,1000],'alpha':[0.0001,0.0005,0.001,0.005,0.01]}#MLP classifier
param_grid5={'penalty':['l1', 'l2', 'elasticnet', 'None'],'solver':['lbfgs' 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],'max_iter':[100,200,400,800,1000,2000,1500],'C':[0.1,0.001,0.0001,0.005,2],'class_weight':['dict','balanced']}#logarithmicRegression
param_grid6={'algorithm':['SAMME', 'SAMME.R']}#Adaboost
param_grid7={'n_estimators':[10,15,16,20,100,200],'criterion':['gini', 'entropy','log_loss'],'max_features':['sqrt', 'log2', 'None'],'ccp_alpha':[0.02,0.01,0.05,0.1],'max_depth':[10,20,30,50,100,200]}#RandomForestClassifier
param_grid8={'var_smoothing':[1e-10,1e-9,1e-8,1e-7,1e-6]}#GaussianNB

print('KNN GridSearchCV')

gv2=GridSearchCV(estimator=KNeighborsClassifier(),param_grid=param_grid2,cv=6,n_jobs=-1,scoring='f1_macro')
gv2.fit(x_train,y_train)
print(gv2.best_params_)


print('SVC GridSearchCV')


gv3=GridSearchCV(estimator=SVC(),param_grid=param_grid3,cv=6,n_jobs=-1,scoring='f1_macro')
gv3.fit(x_train,y_train)
print(gv3.best_params_)

print('MLP GridSearchCV')

gv4=GridSearchCV(estimator=MLPClassifier(),param_grid=param_grid4,cv=6,n_jobs=-1,scoring='f1_macro')
gv4.fit(x_train,y_train)
print(gv4.best_params_)

print('LR GridSearchCV')

gv5=GridSearchCV(estimator=LogisticRegression(),param_grid=param_grid5,cv=6,n_jobs=-1,scoring='f1_macro')
gv5.fit(x_train,y_train)
print(gv5.best_params_)

print('RFC GridSearchCV')

gv7=GridSearchCV(estimator=RandomForestClassifier(),param_grid=param_grid7,cv=6,n_jobs=-1,scoring='f1_macro')
gv7.fit(x_train,y_train)
print(gv7.best_params_)

print('GNB GridSearchCV')

gv8=GridSearchCV(estimator=GaussianNB(),param_grid=param_grid8,cv=6,n_jobs=-1,scoring='f1_macro')
gv8.fit(x_train,y_train)
print(gv8.best_params_)

#Assigning the parameters

gnb=GaussianNB(var_smoothing=1e-07)
svc=SVC(C=40,coef0=0,decision_function_shape='ovo',gamma='scale',kernel='sigmoid',shrinking= True)#updated
knn=KNeighborsClassifier(algorithm='auto',leaf_size=5,n_neighbors= 5,p= 3,weights= 'distance')#updated
mlp=MLPClassifier(activation= 'relu',alpha=0.0005, max_iter= 1000, solver= 'adam')
lr=LogisticRegression(C= 2,class_weight= 'balanced',max_iter= 100,penalty= 'l2',solver='newton-cg')
rfc=RandomForestClassifier(ccp_alpha= 0.01,criterion= 'entropy',max_depth= 10,max_features= 'sqrt',n_estimators= 10)#updated


# ## Fitting and training 



lr.fit(x_train,y_train)
svc.fit(x_train,y_train)
rfc.fit(x_train,y_train)
knn.fit(x_train,y_train)
mlp.fit(x_train,y_train)
gnb.fit(x_train,y_train)


# ## Predicting on x_test


y_predict1=lr.predict(x_test)
y_predict2=svc.predict(x_test)
y_predict4=rfc.predict(x_test)
y_predict5=knn.predict(x_test)
y_predict6=mlp.predict(x_test)
y_predict7=gnb.predict(x_test)


# # Checking accuracy



print('Logistic Regression')
print(confusion_matrix(y_predict1,y_test))
print(classification_report(y_predict1,y_test))




print("Support Vector Machine for Classification")
print(confusion_matrix(y_predict2,y_test))
print(classification_report(y_predict2,y_test))





print('Random Forest Classifier')
print(confusion_matrix(y_predict4,y_test))
print(classification_report(y_predict4,y_test))




print('KNN')
print(confusion_matrix(y_predict5,y_test))
print(classification_report(y_predict5,y_test))


print('MLP')
print(confusion_matrix(y_predict6,y_test))
print(classification_report(y_predict6,y_test))


print('Gaussian NB')
print(confusion_matrix(y_predict7,y_test))
print(classification_report(y_predict7,y_test))


# # Applying adaboost on logistic regression classifier



adb2=AdaBoostClassifier(LogisticRegression(C= 2,class_weight= 'balanced',max_iter= 100,penalty= 'l2',solver='newton-cg'),algorithm= 'SAMME')
adb2.fit(x_train,y_train)
y_predict8=adb2.predict(x_test)

print(' AdaBoost on logistic regression')
print(confusion_matrix(y_predict8,y_test))
print(classification_report(y_predict8,y_test))

