#import matplotlib.pyplot as plt
import pickle as p1
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression  # Importing LinearRegression directly from sklearn.linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import tree


# data = pd.read_csv("./datasets/winequality-white.csv",sep=";")
# train_data=data[:1000]
train_data = pd.read_csv("./datasets/dataset_finalrealdata_handwriting_train.csv",sep=";" ,  on_bad_lines='skip' )
data_X = train_data.iloc[:,0:11]
data_Y1 = train_data.iloc[:,11:12]
data_Y = np.ravel(data_Y1)
#print(train_data.columns)
print(data_X)
print(data_Y)

#colum_train=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']

# Using LinearRegression directly from sklearn.linear_model

clf = tree.DecisionTreeRegressor()
clf_svm_model = clf.fit(data_X , data_Y)
#regr = LinearRegression()
#preditor_linear_model = regr.fit(data_X, data_Y)
preditor_Pickle = open('./handwriting_predictor_model', 'wb')
print("handwriting_predictor_model")
#p1.dump(preditor_linear_model, preditor_Pickle)
p1.dump(clf_svm_model, preditor_Pickle)
rr = clf.score(data_X, data_Y)
print("coef. Correl", rr)
