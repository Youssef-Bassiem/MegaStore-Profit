import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.linear_model import Ridge
from sklearn import metrics


def DateFormat(mega_store):
    mega_store['Order Date'] = pd.to_datetime(mega_store['Order Date'], dayfirst=True)
    mega_store['Order Year'] = mega_store['Order Date'].dt.year
    mega_store['Order Month'] = mega_store['Order Date'].dt.month
    mega_store['Order Day'] = mega_store['Order Date'].dt.day

    mega_store.drop(['Order Date'], axis=1, inplace=True)

    mega_store['Ship Date'] = pd.to_datetime(mega_store['Ship Date'], dayfirst=True)
    mega_store['Ship Year'] = mega_store['Ship Date'].dt.year
    mega_store['Ship Month'] = mega_store['Ship Date'].dt.month
    mega_store['Ship Day'] = mega_store['Ship Date'].dt.day

    mega_store.drop(['Ship Date'], axis=1, inplace=True)
    return mega_store

def outlier(column):
    sorted(column)
    Q1, Q3 = np.percentile(column, [25, 75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range, upper_range


mega_store = pd.read_csv("megastore-regression-dataset.csv")
counts = 0
for i in range(mega_store['Country'].__len__()):
    if mega_store['Country'][i] != 'United States':
        counts += 1

if (counts == 0):
    mega_store = mega_store.drop(['Country' ,'Row ID' ,'Order ID' ,'Customer Name' ,'Customer ID' ,'Product ID'], axis=1)



lowerbound, upperbound = outlier(mega_store['Sales'])
print(lowerbound ,upperbound)

mega_store.drop(mega_store[(mega_store.Sales > upperbound) | (mega_store.Sales < lowerbound)].index, inplace=True)

mega_store = DateFormat(mega_store)

tst = mega_store['CategoryTree'].str.split(',|:', expand=True)
mega_store = mega_store.drop(['CategoryTree'] ,axis=1)
mega_store['MainCategory'] = tst[1].squeeze()
mega_store['SubCategory'] = tst[3].squeeze()

mega_store_encoder = LabelEncoder()
mega_store['Region'] = mega_store_encoder.fit_transform(mega_store['Region'])
mega_store['Segment'] = mega_store_encoder.fit_transform(mega_store['Segment'])
mega_store['City'] = mega_store_encoder.fit_transform(mega_store['City'])
mega_store['State'] = mega_store_encoder.fit_transform(mega_store['State'])
mega_store['Product Name'] = mega_store_encoder.fit_transform(mega_store['Product Name'])
mega_store['Ship Mode'] = mega_store_encoder.fit_transform(mega_store['Ship Mode'])
mega_store['MainCategory'] = mega_store_encoder.fit_transform(mega_store['MainCategory'])
mega_store['SubCategory'] = mega_store_encoder.fit_transform(mega_store['SubCategory'])

corr = mega_store.corr()

scaler = MinMaxScaler()
mega_store = pd.DataFrame(scaler.fit_transform(mega_store) ,columns= mega_store.columns)

Y_Data = mega_store['Profit']
X_Data = mega_store.drop(['Profit'] ,axis = 1)

print(X_Data.shape)
print(corr)

y=mega_store["Profit"]
c=mega_store.drop(['Profit'] ,axis = 1)

#anova

fvalue_Best = SelectKBest(f_classif, k=9)
X_kbest = fvalue_Best.fit_transform(c, y)
print("correlation",X_kbest)



X=mega_store[["Sales","MainCategory"]]
y=mega_store["Profit"]

model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression(fit_intercept = False))

X_train, X_test, Y_train, Y_test = train_test_split(
    X_kbest, y, test_size=0.2, random_state=0)
#fit model
model.fit(X_train, Y_train)
print("Polynomial Result : ",model.score(X_test ,Y_test))
print("Mean Square Error poly", metrics.mean_squared_error(Y_test, model.predict(X_test)))

# saving Polynomial Model
filename = 'Polynomial_Regression_Model.sav'
pickle.dump(model, open(filename, 'wb'))
# Reading the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print("pickle result : ",result)

#ridge
X = mega_store[["Ship Day","Ship Month","Ship Year"]]
y = mega_store["Profit"]
# X = mega_store.drop(['Profit'] ,axis = 1)
X_train, X_test, Y_train, Y_test = train_test_split(
    X_kbest, y, test_size=0.2, random_state=10,shuffle=True)

model = Ridge(alpha=1.0)
model.fit(X_train, Y_train)
v = model.score(X_test, Y_test)
print("Ridge Score : ",v)
print("Mean Square Error ridge", metrics.mean_squared_error(Y_test, model.predict(X_test)))


# saving Ridge Model
filename = 'Ridge_Regression_Model.sav'
pickle.dump(model, open(filename, 'wb'))
# Reading the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print("pickle result : ",result)

