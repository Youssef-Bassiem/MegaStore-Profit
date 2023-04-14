import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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



# #lasso reg
# from sklearn.linear_model import LassoCV
# from sklearn.model_selection import RepeatedKFold
#
# X = mega_store[["Sales","Ship Day"]]
# y = mega_store["Profit"]
#
# #define cross-validation method to evaluate model
# #cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=20)
# #define model
# clf = LassoCV(alphas = np.logspace(-1,1,3), cv=cv, max_iter = 1000000, tol = 0.005)
# #model = LassoCV(alphas=np.arange(0, 1, 0.01), cv=cv)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=0)
# #fit model
# clf.fit(X_train, y_train)
#
# print("Mean Square Error lasso", metrics.mean_squared_error(y_test,  clf.predict(X_test)))


#predict hp value using lasso regression model
# print(clf.predict(X_test))


#polynomial

# def Polynomial(x ,n ,column ,data ,index=0 ,powers=0):
#     if index == x.shape[1]:
#         if powers <= n:
#             data = np.append(data ,column ,axis=1)
#         return data
#
#     for i in range(n+1):
#         col = np.power(x[:,index] ,i).reshape(-1,1)
#         mult = col * column
#         data = Polynomial(x ,n ,mult ,data ,index+1 ,powers+i)
#     return data
#
# from sklearn import linear_model
# X = mega_store[["Sales","Ship Day"]]
# y = mega_store["Profit"]
#
# x_data = Polynomial(X ,2 ,column= np.ones((X.shape[0] ,1)) ,data= np.zeros((X.shape[0] ,1)))
#
# model= linear_model.LinearRegression()
#
# X_train, X_test, y_train, y_test = train_test_split(
#     x_data, y, test_size=0.2, random_state=0)
# #fit model
# model.fit(X_train, y_train)
#
# print("Mean Square Error", metrics.mean_squared_error(y_test,  model.predict(X_test)))
#
# #predict hp value using lasso regression model


from sklearn.linear_model import LinearRegression
#polynomial
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

X=mega_store[["Sales","MainCategory"]]
y=mega_store["Profit"]

model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression(fit_intercept = False))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
#fit model
model.fit(X_train, y_train)

print("Mean Square Error poly", metrics.mean_squared_error(y_test,  model.predict(X_test)))

# print(model.namedsteps.linearregression.coef)
# print(model.predict(X_predict))

#ridge
from sklearn.linear_model import Ridge
X=mega_store[["Ship Day","Ship Month","Ship Year"]]
y=mega_store["Profit"]
# X = mega_store.drop(['Profit'] ,axis = 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10,shuffle=True)

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

print("Mean Square Error ridge", metrics.mean_squared_error(y_test,  model.predict(X_test)))
