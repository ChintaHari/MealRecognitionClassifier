#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.fftpack import fft, ifft,rfft
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from joblib import dump, load


# In[2]:


data=pd.read_csv('test.csv',header=None)


# In[3]:


def createnomealfeaturematrix(non_meal_data):
    index_to_remove_non_meal=non_meal_data.isna().sum(axis=1).replace(0,np.nan).dropna().where(lambda x:x>5).dropna().index
    non_meal_data_cleaned=non_meal_data.drop(non_meal_data.index[index_to_remove_non_meal]).reset_index().drop(columns='index')
    non_meal_data_cleaned=non_meal_data_cleaned.interpolate(method='linear',axis=1)
    index_to_drop_again=non_meal_data_cleaned.isna().sum(axis=1).replace(0,np.nan).dropna().index
    non_meal_data_cleaned=non_meal_data_cleaned.drop(non_meal_data_cleaned.index[index_to_drop_again]).reset_index().drop(columns='index')
    non_meal_feature_matrix=pd.DataFrame()
    #non_meal_data_cleaned['tau_time']=(24-non_meal_data_cleaned.iloc[:,0:19].idxmax(axis=1))*5
    #non_meal_data_cleaned['difference_in_glucose_normalized']=(non_meal_data_cleaned.iloc[:,0:19].max(axis=1)-non_meal_data_cleaned.iloc[:,24])/(non_meal_data_cleaned.iloc[:,24])
    tau_time=(non_meal_data_cleaned.iloc[:,1:].idxmax(axis=1)-0)*5
    difference_in_glucose_normalized=(non_meal_data_cleaned.iloc[:,1:].max(axis=1)-non_meal_data_cleaned.iloc[:,0])/(non_meal_data_cleaned.iloc[:,0])

    power_first_max,index_first_max,power_second_max,index_second_max=[],[],[],[]
    for i in range(len(non_meal_data_cleaned)):
        array=abs(rfft(non_meal_data_cleaned.iloc[:,0:24].iloc[i].values.tolist())).tolist()
        sorted_array=abs(rfft(non_meal_data_cleaned.iloc[:,0:24].iloc[i].values.tolist())).tolist()
        sorted_array.sort()
        power_first_max.append(sorted_array[-2])
        power_second_max.append(sorted_array[-3])
        index_first_max.append(array.index(sorted_array[-2]))
        index_second_max.append(array.index(sorted_array[-3]))
    #non_meal_feature_matrix['tau_time']=non_meal_data_cleaned['tau_time']
    #non_meal_feature_matrix['difference_in_glucose_normalized']=non_meal_data_cleaned['difference_in_glucose_normalized']
    non_meal_feature_matrix['tau_time']=tau_time
    non_meal_feature_matrix['difference_in_glucose_normalized']=difference_in_glucose_normalized
    non_meal_feature_matrix['power_first_max']=power_first_max
    non_meal_feature_matrix['power_second_max']=power_second_max
    non_meal_feature_matrix['index_first_max']=index_first_max
    non_meal_feature_matrix['index_second_max']=index_second_max
    first_differential_data=[]
    second_differential_data=[]
    for i in range(len(non_meal_data_cleaned)):
        first_differential_data.append(np.diff(non_meal_data_cleaned.iloc[:,0:24].iloc[i].tolist()).max())
        second_differential_data.append(np.diff(np.diff(non_meal_data_cleaned.iloc[:,0:24].iloc[i].tolist())).max())
    non_meal_feature_matrix['1stDifferential']=first_differential_data
    non_meal_feature_matrix['2ndDifferential']=second_differential_data
    return non_meal_feature_matrix


# In[4]:


dataset=createnomealfeaturematrix(data)


# In[5]:


from joblib import dump, load
with open('DecisionTreeClassifier.pickle', 'rb') as pre_trained:
    pickle_file = load(pre_trained)
    predict = pickle_file.predict(dataset)    
    pre_trained.close()


# In[6]:


# from joblib import dump, load
# with open('SVC_classifier.joblib', 'rb') as pre_trained:
#     pickle_file = load(pre_trained)
#     predict = pickle_file.predict(dataset)    
#     pre_trained.close()


# In[7]:


pd.DataFrame(predict).to_csv('Result.csv',index=False,header=False)

