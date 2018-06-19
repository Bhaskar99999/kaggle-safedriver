import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.utils import resample
from sklearn.decomposition import PCA

dataset = pd.read_csv('train.csv')
df_majority = dataset[dataset.target==0]
df_minority = dataset[dataset.target==1]
df_minority_upsampled = resample(df_minority, 
                                 replace=True,
                                 n_samples=573518,
                                 random_state=123)
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
X = df_upsampled.iloc[:, 1:58].values
y = df_upsampled.iloc[:, 58].values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = -1, strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X[:, [4,20,21,22,23,25,27,33,35]])
X[:, [4,20,21,22,23,25,27,33,35]] = imputer.transform(X[:, [4,20,21,22,23,25,27,33,35]])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)
#classifier = LogisticRegression(random_state = 0)
#classifier.fit(X_train, y_train)
classifier = XGBClassifier(learning_rate=0.5,max_depth=12,gamma=2,n_estimators=5000,subsample=0.8,objective='binary:logistic',base_score=7.76,min_child_weight=1,colsample_bytree=0.5,reg_alpha=1)
classifier.fit(X_train, y_train)
assifier.score(X_test,y_test)
recall_score(y_test, classifier.predict(X_test))
y_pred = classifier.predict_proba(X_test)
print(roc_auc_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
