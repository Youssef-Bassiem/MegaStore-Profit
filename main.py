import pickle

import pandas as pd
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split

from Preprocessing import preprocessing
from Regression_Models import RegressionModels
from Classification_Models import ClassificationModels

mega_store_regression = pd.read_csv("megastore-regression-dataset.csv")
mega_store_classification = pd.read_csv("megastore-classification-dataset.csv")

# X_Data, Y_Data = preprocessing(mega_store_regression, True, False)
#
# RegressionModels.ridge(X_Data, Y_Data)
#
X_Data, Y_Data = preprocessing(mega_store_classification, False, False)

# ANOVA to get best 10 features for poly model
fvalue_Best = SelectKBest(f_classif, k=10)
X_Data = fvalue_Best.fit_transform(X_Data, Y_Data)
#
# X_train, X_test, Y_train, Y_test = train_test_split(
#     X_Data, Y_Data, test_size=0.2, random_state=10, shuffle=True)
#
# RegressionModels.poly(X_train, Y_train, X_test, Y_test)


X_train, X_test, Y_train, Y_test = train_test_split(
    X_Data, Y_Data, test_size=0.2, random_state=10, shuffle=True)

ClassificationModels.adaboost(X_train, Y_train, X_test, Y_test)
ClassificationModels.tree(X_train, Y_train, X_test, Y_test)
ClassificationModels.naive(X_train, Y_train, X_test, Y_test)

tst = pd.read_csv("megastore-regression-dataset.csv")
X_Data, Y_Data = preprocessing(tst, True, True)

# ANOVA to get best 10 features for poly model
fvalue_Best = SelectKBest(f_classif, k=10)
X_Data = fvalue_Best.fit_transform(X_Data, Y_Data)

# X_train, X_test, Y_train, Y_test = train_test_split(
#     X_Data, Y_Data, test_size=0.9, random_state=10, shuffle=True)

filename = 'Polynomial_Regression_Model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
print("pickle result : ", loaded_model.score(X_Data, Y_Data))
print("Mean Square Error poly", metrics.mean_squared_error(Y_Data, loaded_model.predict(X_Data)))

X_Data = tst[["Ship Mode", "Ship Day", "Ship Year", "Order Year", "Customer ID", "Customer Name", "Order ID",
              "Postal Code"]]
filename = 'Ridge_Regression_Model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
print("pickle result : ", loaded_model.score(X_Data, Y_Data))
print("Mean Square Error Ridge", metrics.mean_squared_error(Y_Data, loaded_model.predict(X_Data)))

tst = pd.read_csv("megastore-classification-dataset.csv")
X_Data, Y_Data = preprocessing(tst, False, True)

# ANOVA to get best 10 features for poly model
fvalue_Best = SelectKBest(f_classif, k=10)
X_Data = fvalue_Best.fit_transform(X_Data, Y_Data)

filename = 'AdaBoost_Classification_Model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
print("pickle result : ", loaded_model.score(X_Data, Y_Data))
print("Mean Square Error AdaBoost", metrics.mean_squared_error(Y_Data, loaded_model.predict(X_Data)))

filename = 'Tree_Classification_Model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
print("pickle result : ", loaded_model.score(X_Data, Y_Data))
print("Mean Square Error Tree", metrics.mean_squared_error(Y_Data, loaded_model.predict(X_Data)))
