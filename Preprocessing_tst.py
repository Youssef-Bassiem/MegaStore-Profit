# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import MinMaxScaler
# import pickle
#
#
# # Drop date column and add 3 columns day, month, year
# def dateformat(df, date, year, month, day):
#     df[date] = pd.to_datetime(df[date])
#     df[year] = df[date].dt.year
#     df[month] = df[date].dt.month
#     df[day] = df[date].dt.day
#
#     df.drop([date], axis=1, inplace=True)
#     return df
#
#
# # Encode to numeric columns
# def encode(df):
#     filename = 'encoder_Region_Model.sav'
#     loaded_model = pickle.load(open(filename, 'rb'))
#     df['Region'] = loaded_model.transform(df['Region'])
#
#     filename = 'encoder_Segment_Model.sav'
#     loaded_model = pickle.load(open(filename, 'rb'))
#     df['Segment'] = loaded_model.transform(df['Segment'])
#
#     filename = 'encoder_City_Model.sav'
#     loaded_model = pickle.load(open(filename, 'rb'))
#     df['City'] = loaded_model.transform(df['City'])
#
#     filename = 'encoder_State_Model.sav'
#     loaded_model = pickle.load(open(filename, 'rb'))
#     df['State'] = loaded_model.transform(df['State'])
#
#     filename = 'encoder_Product_Name_Model.sav'
#     loaded_model = pickle.load(open(filename, 'rb'))
#     df['Product Name'] = loaded_model.transform(df['Product Name'])
#
#     filename = 'encoder_Customer_Name_Model.sav'
#     loaded_model = pickle.load(open(filename, 'rb'))
#     df['Customer Name'] = loaded_model.transform(df['Customer Name'])
#
#     filename = 'encoder_Ship_Mode_Model.sav'
#     loaded_model = pickle.load(open(filename, 'rb'))
#     df['Ship Mode'] = loaded_model.transform(df['Ship Mode'])
#
#     filename = 'encoder_MainCategory_Model.sav'
#     loaded_model = pickle.load(open(filename, 'rb'))
#     df['MainCategory'] = loaded_model.transform(df['MainCategory'])
#
#     filename = 'encoder_SubCategory_Model.sav'
#     loaded_model = pickle.load(open(filename, 'rb'))
#     df['SubCategory'] = loaded_model.transform(df['SubCategory'])
#
#     filename = 'encoder_Country_Model.sav'
#     loaded_model = pickle.load(open(filename, 'rb'))
#     df['Country'] = loaded_model.transform(df['Country'])
#
#     filename = 'encoder_Row_ID_Model.sav'
#     loaded_model = pickle.load(open(filename, 'rb'))
#     df['Row ID'] = loaded_model.transform(df['Row ID'])
#
#     filename = 'encoder_Order_ID_Model.sav'
#     loaded_model = pickle.load(open(filename, 'rb'))
#     df['Order ID'] = loaded_model.transform(df['Order ID'])
#
#     filename = 'encoder_Customer_ID_Model.sav'
#     loaded_model = pickle.load(open(filename, 'rb'))
#     df['Customer ID'] = loaded_model.transform(df['Customer ID'])
#
#     filename = 'encoder_Product_ID_Model.sav'
#     loaded_model = pickle.load(open(filename, 'rb'))
#     df['Product ID'] = loaded_model.transform(df['Product ID'])
#
#     return df
#
#
# # From dict to 2 columns
# def divide(df, main_category):
#     tmp = df[main_category].str.split(',|:', expand=True)
#     df.drop([main_category], axis=1, inplace=True)
#     df['MainCategory'] = tmp[1].squeeze()
#     df['SubCategory'] = tmp[3].squeeze()
#     return df
#
#
# def preprocessing_tst(df, flag):
#     count = 0
#     for i in range(df['Country'].__len__()):
#         if df['Country'][i] != 'United States':
#             count += 1
#
#     df = divide(df, 'CategoryTree')
#     df = encode(df)
#
#     # Date to 3 columns Day, Month, Year
#     df = dateformat(df, 'Order Date', 'Order Year', 'Order Month', 'Order Day')
#     df = dateformat(df, 'Ship Date', 'Ship Year', 'Ship Month', 'Ship Day')
#
#     if count == 0:
#         df.drop(['Country', 'Ship Month', 'Order Month', 'Order Day', 'Row ID', 'Product Name'],
#                 axis=1, inplace=True)
#
#     scaler = MinMaxScaler()
#     if flag:
#         df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
#         return df.drop(['Profit'], axis=1), df['Profit']
#     else:
#         y_data = df['ReturnCategory']
#         df.drop(['ReturnCategory'], inplace=True, axis=1)
#     df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
#
#     return df, y_data
