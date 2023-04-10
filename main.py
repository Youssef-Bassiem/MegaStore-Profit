import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


mega_store = pd.read_csv("megastore-regression-dataset.csv")
counts = 0
for i in range(mega_store['Country'].__len__()):
    if mega_store['Country'][i] != 'United States':
        counts += 1

if (counts == 0):
    mega_store = mega_store.drop(['Country' ,'Row ID' ,'Order ID' ,'Customer Name' ,'Customer ID' ,'Product ID'], axis=1)

# print(mega_store.shape)
# mega_store = mega_store.drop_duplicates(subset=['Product Name'])
# print(mega_store.shape)

mega_store_encoder = LabelEncoder()
mega_store['Region'] = mega_store_encoder.fit_transform(mega_store['Region'])
mega_store['Segment'] = mega_store_encoder.fit_transform(mega_store['Segment'])
mega_store['City'] = mega_store_encoder.fit_transform(mega_store['City'])
mega_store['State'] = mega_store_encoder.fit_transform(mega_store['State'])
mega_store['Product Name'] = mega_store_encoder.fit_transform(mega_store['Product Name'])

corr = mega_store.corr()
print(mega_store['Sales'].max())
print(mega_store['Sales'].min())
