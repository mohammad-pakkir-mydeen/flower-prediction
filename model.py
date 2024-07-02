import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split


df=pd.read_csv("iris.csv")
print(df.head())

X=df.drop(['variety'],axis=1)
y=df['variety']

trainx,testx,trainy,testy=train_test_split(X,y,random_state=10,test_size=0.2)

sc=StandardScaler()
trainx=sc.fit_transform(trainx)
testx=sc.transform(testx)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(trainx,trainy)

import pickle
pickle.dump(classifier,open("model.pkl","wb"))