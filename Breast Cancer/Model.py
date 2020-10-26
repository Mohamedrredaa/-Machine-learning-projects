'''
This code was developed to learn from a dataset which was captured from 
digitlized feature extractor of a cell from breast to find out wethere the 
cancer is malignant or benign
Developed in 26/10/2020
developed by /Mohamed Reda 
'''


import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def FeatureScaling(X_train,X_test):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test


dataset = pd.read_csv('data.csv')
X = dataset.iloc[:,2:]
y = dataset.iloc[:,1]
X = X.drop(['Unnamed: 32'], axis=1)

lb = LabelEncoder()
y  = lb.fit_transform(y) # M = 1 and B = 0

#splitting the data into training and testing sets 
X_train, X_test , y_train , y_test = train_test_split(X,y, test_size = 0.2 , random_state=42)
X_train, X_test = FeatureScaling(X_train , X_test)

#Try to find the correlation between features and labels to best tune a model
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explain_variance = pca.explained_variance_ratio_ 

#building random forest model 
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=15 ,random_state=0 )
classifier.fit(X_train , y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
con_mat_Random_Forest = confusion_matrix(y_true= y_test ,y_pred= y_pred)
print("*"*70)
print("Accuracy Random Forest= " + str(round(((con_mat_Random_Forest[0][0]+con_mat_Random_Forest[1][1]) / len(y_test))*100,2))+"%")

from sklearn.model_selection  import cross_val_score
print(cross_val_score(classifier, X_train, y_train, cv=10))













