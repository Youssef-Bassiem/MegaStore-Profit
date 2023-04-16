import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.linear_model import Ridge
from sklearn import metrics


# Drop date column and add 3 columns day, month, year
def dateformat(df, date, year, month, day):
    df[date] = pd.to_datetime(df[date])
    df[year] = df[date].dt.year
    df[month] = df[date].dt.month
    df[day] = df[date].dt.day

    df.drop([date], axis=1, inplace=True)
    return df


# Get outliers of column using IQR method
def outlier(df, column):
    q1, q3 = np.percentile(sorted(df[column]), [25, 75])
    iqr = q3 - q1
    lower_range = q1 - (1.5 * iqr)
    upper_range = q3 + (1.5 * iqr)
    df.drop(df[(df[column] > upper_range) | (df[column] < lower_range)].index, inplace=True)
    return df


mega_store = pd.read_csv("megastore-regression-dataset.csv")

counts = 0
for i in range(mega_store['Country'].__len__()):
    if mega_store['Country'][i] != 'United States':
        counts += 1

# From dict to 2 columns
tmp = mega_store['CategoryTree'].str.split(',|:', expand=True)
mega_store.drop(['CategoryTree'], axis=1, inplace=True)
mega_store['MainCategory'] = tmp[1].squeeze()
mega_store['SubCategory'] = tmp[3].squeeze()

# Encode to numeric columns
mega_store_encoder = LabelEncoder()
mega_store['Region'] = mega_store_encoder.fit_transform(mega_store['Region'])
mega_store['Segment'] = mega_store_encoder.fit_transform(mega_store['Segment'])
mega_store['City'] = mega_store_encoder.fit_transform(mega_store['City'])
mega_store['State'] = mega_store_encoder.fit_transform(mega_store['State'])
mega_store['Product Name'] = mega_store_encoder.fit_transform(mega_store['Product Name'])
mega_store['Customer Name'] = mega_store_encoder.fit_transform(mega_store['Customer Name'])
mega_store['Ship Mode'] = mega_store_encoder.fit_transform(mega_store['Ship Mode'])
mega_store['MainCategory'] = mega_store_encoder.fit_transform(mega_store['MainCategory'])
mega_store['SubCategory'] = mega_store_encoder.fit_transform(mega_store['SubCategory'])
mega_store['Country'] = mega_store_encoder.fit_transform(mega_store['Country'])
mega_store['Row ID'] = mega_store_encoder.fit_transform(mega_store['Row ID'])
mega_store['Order ID'] = mega_store_encoder.fit_transform(mega_store['Order ID'])
mega_store['Customer ID'] = mega_store_encoder.fit_transform(mega_store['Customer ID'])
mega_store['Product ID'] = mega_store_encoder.fit_transform(mega_store['Product ID'])

# Date to 3 columns Day, Month, Year
mega_store = dateformat(mega_store, 'Order Date', 'Order Year', 'Order Month', 'Order Day')
mega_store = dateformat(mega_store, 'Ship Date', 'Ship Year', 'Ship Month', 'Ship Day')

# Drop unnecessary column and lowest 5 correlation
corr = abs(mega_store.corr(numeric_only=True))

if counts == 0:
    mega_store = mega_store.drop(['Country', 'Ship Month', 'Order Month', 'Order Day', 'Row ID', 'Product Name'],
                                 axis=1)

mega_store = outlier(mega_store, 'Sales')
mega_store = outlier(mega_store, 'Discount')
# mega_store = outlier(mega_store, 'MainCategory')
mega_store = outlier(mega_store, 'Product ID')
mega_store = outlier(mega_store, 'Quantity')
mega_store = outlier(mega_store, 'SubCategory')
mega_store = outlier(mega_store, 'Postal Code')
mega_store = outlier(mega_store, 'Region')
mega_store = outlier(mega_store, 'State')
mega_store = outlier(mega_store, 'City')
mega_store = outlier(mega_store, 'Segment')
mega_store = outlier(mega_store, 'Order Year')
mega_store = outlier(mega_store, 'Ship Year')
mega_store = outlier(mega_store, 'Ship Day')
mega_store = outlier(mega_store, 'Ship Mode')
mega_store = outlier(mega_store, 'Order ID')
mega_store = outlier(mega_store, 'Customer ID')
mega_store = outlier(mega_store, 'Customer Name')

corr = abs(mega_store.corr())

scaler = MinMaxScaler()
mega_store = pd.DataFrame(scaler.fit_transform(mega_store), columns=mega_store.columns)

Y_Data = mega_store['Profit']
X_Data = mega_store.drop(['Profit'], axis=1)

# ANOVA to get best 10 features for poly model
fvalue_Best = SelectKBest(f_classif, k=10)
X_Data = fvalue_Best.fit_transform(X_Data, Y_Data)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_Data, Y_Data, test_size=0.2, random_state=10, shuffle=True)

poly_model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression(fit_intercept=False))

# fit model
poly_model.fit(X_train, Y_train)
print("Polynomial Score : ", poly_model.score(X_test, Y_test))
print("Mean Square Error poly", metrics.mean_squared_error(Y_test, poly_model.predict(X_test)))

# saving Polynomial Model
filename = 'Polynomial_Regression_Model.sav'
pickle.dump(poly_model, open(filename, 'wb'))

# Reading the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
print("pickle result : ", loaded_model.score(X_test, Y_test))

# ridge with bad 8 features
X = mega_store[["Ship Mode", "Ship Day", "Ship Year", "Order Year", "Customer ID", "Customer Name", "Order ID",
                "Postal Code"]]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_Data, test_size=0.2, random_state=10, shuffle=True)

# clc = LinearRegression()
# clc.fit(X_train, Y_train)
# print("cls Score : ", clc.score(X_test, Y_test))

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, Y_train)

print("Ridge Score : ", ridge_model.score(X_test, Y_test))
print("Mean Square Error ridge", metrics.mean_squared_error(Y_test, ridge_model.predict(X_test)))

# saving Ridge Model
filename = 'Ridge_Regression_Model.sav'
pickle.dump(ridge_model, open(filename, 'wb'))

# Reading the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
print("pickle result : ", loaded_model.score(X_test, Y_test))
