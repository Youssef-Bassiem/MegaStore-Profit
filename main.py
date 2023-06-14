import pickle
import pandas as pd
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split

from Classification_Models import ClassificationModels
from Preprocessing import preprocessing
from Regression_Models import RegressionModels

tst = int(input("Train : 0, Test : 1\n"))
flag = int(input("Regression : 0, Classification : 1\n"))

# RegressionModels Training
#####################################################################################
if not tst and not flag:
    mega_store_regression = pd.read_csv("megastore-regression-dataset.csv")
    X_Data, Y_Data = preprocessing(mega_store_regression, True, False)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_Data, Y_Data, test_size=0.2, random_state=10, shuffle=True)

    # ANOVA to get best 10 features for poly model
    fvalue_Best = SelectKBest(f_classif, k=10)
    fvalue_Best.fit_transform(X_train, Y_train)

    # Get columns to keep and create new dataframe with those only
    cols_index = fvalue_Best.get_support(indices=True)
    filename = 'Models/Features_selection/fvalue_Best_regression_Model.sav'
    pickle.dump(fvalue_Best, open(filename, 'wb'))

    X_test = X_test.iloc[:, cols_index]
    X_train = X_train.iloc[:, cols_index]

    RegressionModels.ridge(X_train, Y_train, X_test, Y_test)

    RegressionModels.poly(X_train, Y_train, X_test, Y_test)
#####################################################################################

# RegressionModels Load (Test)
#####################################################################################
elif tst and not flag:
    tst = pd.read_csv("megastore-tas-test-regression.csv")
    # tst = pd.read_csv("megastore-regression-dataset.csv")
    X_Data, Y_Data = preprocessing(tst, True, True)

    # Reading the model from disk
    filename = 'Models/Features_selection/fvalue_Best_regression_Model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    loaded_model.transform(X_Data)

    cols_index = loaded_model.get_support(indices=True)
    X_Data = X_Data.iloc[:, cols_index]

    filename = 'Models/Regression/Polynomial_Regression_Model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    print("pickle result poly : ", loaded_model.score(X_Data, Y_Data))
    print("Mean Square Error poly", metrics.mean_squared_error(Y_Data, loaded_model.predict(X_Data)))

    filename = 'Models/Regression/Ridge_Regression_Model.sav'

    loaded_model = pickle.load(open(filename, 'rb'))

    print("pickle result Ridge : ", loaded_model.score(X_Data, Y_Data))
    print("Mean Square Error Ridge", metrics.mean_squared_error(Y_Data, loaded_model.predict(X_Data)))
####################################################################################

# ClassificationModels Training
#####################################################################################
elif not tst and flag:
    mega_store_classification = pd.read_csv("megastore-classification-dataset.csv")

    X_Data, Y_Data = preprocessing(mega_store_classification, False, False)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_Data, Y_Data, test_size=0.2, random_state=10, shuffle=True)

    # ANOVA to get best 10 features for poly model
    fvalue_Best = SelectKBest(f_classif, k=10)
    fvalue_Best.fit_transform(X_train, Y_train)

    # Get columns to keep and create new dataframe with those only
    cols_index = fvalue_Best.get_support(indices=True)

    X_test = X_test.iloc[:, cols_index]
    X_train = X_train.iloc[:, cols_index]

    filename = 'Models/Features_selection/fvalue_Best_classification_Model.sav'
    pickle.dump(fvalue_Best, open(filename, 'wb'))

    ClassificationModels.adaboost(X_train, Y_train, X_test, Y_test)
    ClassificationModels.tree(X_train, Y_train, X_test, Y_test)
    ClassificationModels.naive(X_train, Y_train, X_test, Y_test)
####################################################################################

# ClassificationModels Load (Test)
#####################################################################################
elif tst and flag:
    tst = pd.read_csv("megastore-tas-test-classification.csv")
    # tst = pd.read_csv("megastore-classification-dataset.csv")
    X_Data, Y_Data = preprocessing(tst, False, True)

    # Reading the model from disk
    filename = 'Models/Features_selection/fvalue_Best_classification_Model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    loaded_model.transform(X_Data)

    cols_index = loaded_model.get_support(indices=True)
    X_Data = X_Data.iloc[:, cols_index]

    filename = 'Models/Classification/AdaBoost_Classification_Model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    print("pickle result : ", loaded_model.score(X_Data, Y_Data))
    print("Mean Square Error AdaBoost", metrics.mean_squared_error(Y_Data, loaded_model.predict(X_Data)))

    filename = 'Models/Classification/Tree_Classification_Model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    print("pickle result : ", loaded_model.score(X_Data, Y_Data))
    print("Mean Square Error Tree", metrics.mean_squared_error(Y_Data, loaded_model.predict(X_Data)))

    filename = 'Models/Classification/Naive_Classification_Model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    print("pickle result : ", loaded_model.score(X_Data, Y_Data))
    print("Mean Square Error Naive", metrics.mean_squared_error(Y_Data, loaded_model.predict(X_Data)))
#####################################################################################
