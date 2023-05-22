import pickle

import pandas as pd
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split

from Classification_Models import ClassificationModels
from Preprocessing import preprocessing
from Regression_Models import RegressionModels



flag = int(input("Regression -> 1, Classification -> 2\n"))

if flag == 1:
    mega_store = pd.read_csv("megastore-regression-dataset.csv")
    if 'Profit' in mega_store.columns:
        mega_store.drop(['Profit'], axis = 1, inplace = True) 
    X_Data = preprocessing(mega_store, True, True)

    # Reading the model from disk
    filename = 'Models/Features_selection/fvalue_Best_regression_Model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    loaded_model.transform(X_Data)
    cols_index = loaded_model.get_support(indices=True)
    X_Data = X_Data.iloc[:, cols_index]

    filename = 'Models/Regression/Polynomial_Regression_Model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    print(pd.DataFrame(loaded_model.predict(X_Data)).head(5))

    filename = 'Models/Regression/Ridge_Regression_Model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    print(pd.DataFrame(loaded_model.predict(X_Data)).head(5))
    

elif flag == 2:
    
    mega_store = pd.read_csv("megastore-classification-dataset.csv")
    if 'ReturnCategory' in mega_store.columns:
        mega_store.drop(['ReturnCategory'], axis = 1, inplace = True) 

    X_Data = preprocessing(mega_store, False, True)
    # Reading the model from disk
    filename = 'Models/Features_selection/fvalue_Best_classification_Model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    loaded_model.transform(X_Data)
    cols_index = loaded_model.get_support(indices=True)
    X_Data = X_Data.iloc[:, cols_index]

    filename = 'Models/Encoding/encoder_ReturnCategory_Model.sav'
    encoder = pickle.load(open(filename, 'rb'))

    filename = 'Models/Classification/AdaBoost_Classification_Model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    print(pd.DataFrame(encoder.inverse_transform(loaded_model.predict(X_Data))).head(5))

    filename = 'Models/Classification/Tree_Classification_Model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    print(pd.DataFrame(encoder.inverse_transform(loaded_model.predict(X_Data))).head(5))
    
    filename = 'Models/Classification/Naive_Classification_Model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    print(pd.DataFrame(encoder.inverse_transform(loaded_model.predict(X_Data))).head(5))
