import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

parkinsons_df = pd.read_csv("parkinsons.csv")

X = parkinsons_df.drop(['name', 'status'], axis=1)
y = parkinsons_df['status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
svm = SVC(kernel='rbf', gamma='scale')
parameters = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10, 100]}
clf = GridSearchCV(svm, parameters, cv=5)
clf.fit(X_train, y_train)

print("Best hyperparameters -", clf.best_params_)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy -", accuracy)