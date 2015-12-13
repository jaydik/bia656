
# coding: utf-8

# # Imports

# In[39]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ### Data Import 

# In[40]:

train = pd.read_csv('../data/train.csv')


# In[ ]:




# ## Trim off the zero sales

# In[41]:

non_zero_sales_mask = train['Sales'] > 0

nz_train = train.ix[non_zero_sales_mask, :]
nz_train.ix[:, 'LogSales'] = np.log(nz_train['Sales'])


# In[42]:

nz_train.head()


# In[43]:

for x in ['Open', 'Promo', 'StateHoliday', 'SchoolHoliday']:
    print x, "=", np.unique(nz_train[x])


# In[44]:

print(len(train), len(nz_train))


# ### Log Transforming Sales

# In[45]:


get_ipython().magic(u'pylab inline')
_, _, _ = plt.hist(nz_train.ix[non_zero_sales_mask, 'Sales'])


plt.figure()
_, _, _ = plt.hist(nz_train.ix[non_zero_sales_mask, 'LogSales'])


# ## Add Weeknumber (most predictive date field)

# In[46]:

import datetime as dt

nz_train['WeekNumber'] = nz_train['Date'].apply(lambda x: pd.to_datetime(x).isocalendar()[1])


# ## Encode State Holiday

# In[47]:

from sklearn.preprocessing import LabelEncoder
nz_train.ix[:, ['StateHoliday']] = nz_train.ix[:, ['StateHoliday']].astype(str)

le = LabelEncoder().fit(np.unique(nz_train['StateHoliday']))

nz_train.ix[:, 'StateHolidayTransform'] = le.transform(nz_train.ix[:, 'StateHoliday'])


# In[48]:

nz_train.head()


# In[49]:

np.unique(nz_train['StateHolidayTransform'])


# ## Encode the Store Data

# In[50]:

store = pd.read_csv('../data/store.csv')


# In[51]:

store.head()


# In[52]:

store.ix[:, 'StoreTypeTransformed'] = LabelEncoder().fit_transform(store['StoreType'])


# In[53]:

store.ix[:, 'AssortmentTransformed'] = LabelEncoder().fit_transform(store['Assortment'])


# ## Attempt to Join

# In[ ]:




# In[54]:

joined = nz_train.merge(store, how='inner')


# In[55]:

joined.dtypes


# ## OneHotEconder

# In[56]:

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder().fit(joined.ix[:, ['Promo',
                                        'StateHolidayTransform', 
                                        'SchoolHoliday',
                                        'StoreTypeTransformed',
                                        'AssortmentTransformed']])

categoricals= enc.transform(joined.ix[:, ['Promo',
                                          'StateHolidayTransform', 
                                          'SchoolHoliday',
                                          'StoreTypeTransformed',
                                          'AssortmentTransformed']]).toarray().astype(float64)

training_df = pd.DataFrame(categoricals)
training_df['LogSales'] = joined['LogSales']
training_df['DayOfWeek'] = joined['DayOfWeek']
training_df['Store'] = joined['Store']
training_df['WeekNumber'] = joined['WeekNumber']
training_df['Customers'] = joined['Customers']
training_df['Date'] = joined['Date']


# In[ ]:




# In[ ]:




# In[57]:

training_df.describe()


# In[ ]:




# In[58]:

#traindata2 = pd.merge(training_df, result2, on=['MoDay', 'Store'])


# In[59]:

training_df.head()


# In[60]:

#create a datetime field so i can split it up into Month-Day. MoDay and Store will be a composite key
training_df['Date']= pd.to_datetime(training_df['Date'], format="%Y-%m-%d")
#training_df['MoDay']= training_df['Date'].dt.strftime('%m-%d')


# In[61]:

training_df.describe()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[62]:

decomp = training_df


# In[63]:

decomp['Date']= pd.to_datetime(decomp['Date'], format="%Y-%m-%d")


# In[64]:

decomp = decomp.set_index(pd.DatetimeIndex(decomp['Date']))



# In[65]:

#interpolate missing values
#decomp['LogSales'].interpolate(inplace=True)


# In[75]:

#decompose time series
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(decomp['LogSales'], model='additive', freq=12)  
decomposition2 = seasonal_decompose(decomp['Customers'], model='additive', freq=12)


# In[37]:

#%pylab inline
#fig = plt.figure()  
#fig = decomposition.plot()  


# In[ ]:




# In[76]:

result = pd.concat([decomposition.observed, decomposition.trend], axis=1)
result.columns = ['Observed','Trend']
resultCustomers = pd.concat([decomposition.observed], axis=1)
resultCustomers.columns = ['MeanCustomerObserved']


# In[78]:

#fill missing values
#result = result.fillna(0)
result['Store'] = decomp['Store']
resultCustomers['Store'] = decomp['Store']


# In[79]:


result2 = result.groupby(['Store']).mean()
result2['Store'] = result2.index
resultCustomers2 = resultCustomers.groupby(['Store']).mean()
resultCustomers2['Store'] = resultCustomers2.index
#result2['DayOfWeek'] = result2.index.get_level_values(0)
#result2['Store'] = result2.index.get_level_values(1)

#use Month-Day - it didnt work
#result['MoDay'] = result.index.strftime("%m-%d")
#result2 = result.groupby(['MoDay','Store']).mean()
#result2 = result.groupby(['MoDay','Store']).mean()
#result2['MoDay'] = result2.index.get_level_values(0)
#result2['Store'] = result2.index.get_level_values(1)


# In[ ]:




# In[ ]:




# In[70]:

#mean observed, seasonal, trend, resid for each store, per day
#traindata2.head()


# In[90]:

#create a new training df by merging the mean results with the originail df
#traindata2 = pd.merge(training_df, result2, on=['MoDay', 'Store'])
#traindata2 = pd.merge(training_df, result2, on=['Store'])
traindata2 = pd.merge(training_df,result2,how='left',on=['Store'])
traindata3 = pd.merge(traindata2,resultCustomers2,how='left',on=['Store'])


# In[91]:

traindata3.describe()


# In[92]:

traindata3 = traindata3.drop(['Customers','Date'],axis=1)


# In[93]:

traindata3.dtypes


# In[193]:

#merge result3 with training set


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[194]:

#important_weeks = [18,19,20,21,22,23,24,25,26,27,28,29,30,31]


# In[195]:

#customers_means = training_df[training_df['WeekNumber' == important_weeks]].groupby('Store').mean().Customers
#customers_means.name = 'CustomersMean'



# In[196]:

#training_df = training_df.join(customers_means, on='Store')


# In[120]:

# training_df = joined.ix[:, ['Promo',
#                             'Promo2',
#                             'StateHolidayTransform', 
#                             'SchoolHoliday',
#                             'StoreTypeTransformed',
#                             'AssortmentTransformed',
#                             'LogSales',
#                             'DayOfWeek',
#                             'Store',
#                             'WeekNumber']]

traindata3.to_csv('../cleaneddata/cleanedTraining.csv')


# ## Make Evaluation Set

# In[95]:

test = pd.read_csv('../data/test.csv')


# In[96]:

test.head()


# ## Fix the unknown opens

# In[97]:

test.ix[test['Open'] != 0, 'Open'] = 1


# In[98]:

test.describe()


# In[99]:

test['WeekNumber'] = test['Date'].apply(lambda x: pd.to_datetime(x).isocalendar()[1])
test.ix[:, ['StateHoliday']] = test.ix[:, ['StateHoliday']].astype(str)

test.ix[:, 'StateHolidayTransform'] = LabelEncoder().fit_transform(test['StateHoliday'])



# In[100]:

joined_test = test.merge(store, how='inner')
joined_test.head()


# In[ ]:




# In[101]:


#joined_test['Date']= pd.to_datetime(joined_test['Date'], format="%Y-%m-%d")
#joined_test['MoDay']= joined_test['Date'].dt.strftime('%m-%d')


# In[ ]:




# In[ ]:




# In[102]:

joined_test.head()


# In[103]:

categoricals = enc.transform(joined_test.ix[:, ['Promo',
                                                'StateHolidayTransform', 
                                                'SchoolHoliday',
                                                'StoreTypeTransformed',
                                                'AssortmentTransformed']]).toarray().astype(float64)
testing_df = pd.DataFrame(categoricals)
testing_df['DayOfWeek'] = joined_test['DayOfWeek']
testing_df['Store'] = joined_test['Store']
testing_df['WeekNumber'] = joined_test['WeekNumber']
testing_df['Open'] = joined_test['Open']
testing_df['Id'] = joined_test['Id']
#testing_df['Date'] = joined_test['Date']


# In[104]:

#date time manipulation - MoDay and Store will be used as a composite key
#testing_df['Date']= pd.to_datetime(testing_df['Date'], format="%Y-%m-%d")
#testing_df['MoDay']= testing_df['Date'].dt.strftime('%m-%d')
#testing_df['Store2'] = testing_df['Store']


# In[105]:

testing_df.dtypes


# In[111]:

result2.tail()


# In[108]:

#result2 is the mean of the decomposed time series, per month-day
#missing values - have to use outer join

#testing_df2 = pd.merge(testing_df,result2,how='left',on=['MoDay','Store'])
testing_df2 = pd.merge(testing_df,result2,how='left',on=['Store'])


# In[115]:

testing_df3 = pd.merge(testing_df2,resultCustomers2,how='left',on=['Store'])


# In[116]:

testing_df3.head()


# In[154]:

#testing_df3 = testing_df2[np.isfinite(testing_df2['Store2'])]


# In[112]:

#double check the join because i screwed it up


# In[164]:

#testing_df3 = testing_df3.fillna(testing_df3.mean())


# In[165]:

#result2[(result2['Store']==13) & (result2['MoDay']=="09-08")]


# In[117]:

testing_df3[testing_df3['Observed'].isnull()==True]


# In[121]:

testing_df3[2000:2001]


# In[ ]:




# In[119]:

testing_df3.to_csv('../cleaneddata/cleanedtest.csv')


# In[ ]:


