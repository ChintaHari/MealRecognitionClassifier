#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
from scipy.fftpack import fft, ifft,rfft
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.svm import SVC
from joblib import dump, load


# In[2]:


#insulin_data_df=pd.read_csv('InsulinData.csv',low_memory=False,usecols=['Date','Time','BWZ Carb Input (grams)'])
insulin_data=pd.read_csv('InsulinData.csv',low_memory=False,usecols=['Date','Time','BWZ Carb Input (grams)'])
# insulin_data_df=pd.read_csv('~\InsulinData.csv',low_memory=False,usecols=['Date','Time','BWZ Carb Input (grams)'])


# In[3]:


cgm_data=pd.read_csv('CGMData.csv',low_memory=False,usecols=['Date','Time','Sensor Glucose (mg/dL)'])
# cgm_data_df=pd.read_csv('~\CGMData.csv',low_memory=False,usecols=['Date','Time','Sensor Glucose (mg/dL)'])


# In[4]:


insulin_data['date_time_stamp']=pd.to_datetime(insulin_data['Date'] + ' ' + insulin_data['Time'])
cgm_data['date_time_stamp']=pd.to_datetime(cgm_data['Date'] + ' ' + cgm_data['Time'])


# In[5]:


insulin_data_1=pd.read_csv('Insulin_patient2.csv',low_memory=False,usecols=['Date','Time','BWZ Carb Input (grams)'])
# insulin_data_df_1=pd.read_csv('~\Insulin_patient2.csv',low_memory=False,usecols=['Date','Time','BWZ Carb Input (grams)'])
#print(insulin_data_df_1)


# In[6]:


cgm_data_1=pd.read_csv('CGM_patient2.csv',low_memory=False,usecols=['Date','Time','Sensor Glucose (mg/dL)'])
# cgm_data_df_1=pd.read_csv('~\CGM_patient2.csv',low_memory=False,usecols=['Date','Time','Sensor Glucose (mg/dL)'])
#print(cgm_data_df_1)


# In[7]:


insulin_data_1['date_time_stamp']=pd.to_datetime(insulin_data_1['Date'] + ' ' + insulin_data_1['Time'])
cgm_data_1['date_time_stamp']=pd.to_datetime(cgm_data_1['Date'] + ' ' + cgm_data_1['Time'])

#print(insulin_data_df_1)
#print(cgm_data_df_1)


# # Generating Meal Data using 'createmealdata' function 

# In[8]:


def createmealdata(insulin_data,cgm_data,dateidentifier):
    insulin_df=insulin_data.copy()
    insulin_df=insulin_df.set_index('date_time_stamp')
    valid_2_5_hour_stretch=insulin_df.sort_values(by='date_time_stamp',ascending=True).dropna().reset_index()
    
    valid_2_5_hour_stretch['BWZ Carb Input (grams)'].replace(0.0,np.nan,inplace=True)
    valid_2_5_hour_stretch=valid_2_5_hour_stretch.dropna()
    valid_2_5_hour_stretch=valid_2_5_hour_stretch.reset_index().drop(columns='index')
    valid_timestamp_list=[]
    value=0
    for idx,i in enumerate(valid_2_5_hour_stretch['date_time_stamp']):
        try:
            #print(idx,i)
            value=(valid_2_5_hour_stretch['date_time_stamp'][idx+1]-i).seconds / 60.0
            if value >= 120:
                valid_timestamp_list.append(i)
        except KeyError:
            break
            
            
    #print(valid_timestamp_list[11])
    meal_list=[]
    if dateidentifier==1:
        for idx,i in enumerate(valid_timestamp_list):
            #print(idx,i)
            start=pd.to_datetime(i - timedelta(minutes=30))
            end=pd.to_datetime(i + timedelta(minutes=120))
            get_date=i.date().strftime('%#m/%#d/%Y')
            #print(i, start, end, get_date)
            meal_list.append(cgm_data.loc[cgm_data['Date']==get_date].set_index('date_time_stamp').between_time(start_time=start.strftime('%#H:%#M:%#S'),end_time=end.strftime('%#H:%#M:%#S'))['Sensor Glucose (mg/dL)'].values.tolist())
            #print(meal_list)
        return pd.DataFrame(meal_list)
    else:
        for idx,i in enumerate(valid_timestamp_list):
            start=pd.to_datetime(i - timedelta(minutes=30))
            end=pd.to_datetime(i + timedelta(minutes=120))
            get_date=i.date().strftime('%Y-%m-%d')
            meal_list.append(cgm_data.loc[cgm_data['Date']==get_date].set_index('date_time_stamp').between_time(start_time=start.strftime('%H:%M:%S'),end_time=end.strftime('%H:%M:%S'))['Sensor Glucose (mg/dL)'].values.tolist())
        return pd.DataFrame(meal_list)


# In[9]:




meal_data=createmealdata(insulin_data,cgm_data,1)
meal_data1=createmealdata(insulin_data_1,cgm_data_1,2)
meal_data=meal_data.iloc[:,0:30]
meal_data1=meal_data1.iloc[:,0:30]


# # Generating NoMeal Data using 'createnomealdata' function 

# In[10]:


def createnomealdata(insulin_data,cgm_data):
    insulin_no_meal_df=insulin_data.copy()
    insulin_no_meal_sorted_df=insulin_no_meal_df.sort_values(by='date_time_stamp',ascending=True).replace(0.0,np.nan).dropna().copy()
    insulin_no_meal_sorted_df=insulin_no_meal_sorted_df.reset_index().drop(columns='index')
    valid_timestamp=[]
    for idx,i in enumerate(insulin_no_meal_sorted_df['date_time_stamp']):
        try:
            value=(insulin_no_meal_sorted_df['date_time_stamp'][idx+1]-i).seconds//3600
            #print(value)
            if value >=4:
                valid_timestamp.append(i)
        except KeyError:
            break
    dataset=[]
    for idx, i in enumerate(valid_timestamp):
        iteration_dataset=1
        try:
            length_of_24_dataset_df_count=len(cgm_data.loc[(cgm_data['date_time_stamp']>=valid_timestamp[idx]+pd.Timedelta(hours=2))&(cgm_data['date_time_stamp']<valid_timestamp[idx+1])])//24
            #print(length_of_24_dataset_df_count)
            while (iteration_dataset<=length_of_24_dataset_df_count):
                if iteration_dataset==1:
                    dataset.append(cgm_data.loc[(cgm_data['date_time_stamp']>=valid_timestamp[idx]+pd.Timedelta(hours=2))&(cgm_data['date_time_stamp']<valid_timestamp[idx+1])]['Sensor Glucose (mg/dL)'][:iteration_dataset*24].values.tolist())
                    iteration_dataset+=1
                else:
                    dataset.append(cgm_data.loc[(cgm_data['date_time_stamp']>=valid_timestamp[idx]+pd.Timedelta(hours=2))&(cgm_data['date_time_stamp']<valid_timestamp[idx+1])]['Sensor Glucose (mg/dL)'][(iteration_dataset-1)*24:(iteration_dataset)*24].values.tolist())
                    iteration_dataset+=1
        except IndexError:
            break
    return pd.DataFrame(dataset)


# In[11]:


no_meal_data=createnomealdata(insulin_data,cgm_data)
no_meal_data1=createnomealdata(insulin_data_1,cgm_data_1)


# # Extracting feature matrix for Meal Data

# In[12]:


def createmealfeaturematrix(meal_data):
    #collect indexes where meal_data have more than 6 missing values (i.e nan)
    index=meal_data.isna().sum(axis=1).replace(0,np.nan).dropna().where(lambda x:x>6).dropna().index
    
    meal_data_cleaned=meal_data.drop(meal_data.index[index]).reset_index().drop(columns='index')
    meal_data_cleaned=meal_data_cleaned.interpolate(method='linear',axis=1)
    #print(meal_data_cleaned)
    
    index_to_drop_again=meal_data_cleaned.isna().sum(axis=1).replace(0,np.nan).dropna().index
    meal_data_cleaned=meal_data_cleaned.drop(meal_data.index[index_to_drop_again]).reset_index().drop(columns='index')
    
    #meal_data_cleaned['tau_time']=(meal_data_cleaned.iloc[:,22:25].idxmin(axis=1)-meal_data_cleaned.iloc[:,6].idxmax(axis=1))*5
    #meal_data_cleaned['tau_time']=(meal_data_cleaned.iloc[:,7:].idxmax(axis=1)-6)*5
    tau_time=abs((meal_data_cleaned.iloc[:,7:].idxmax(axis=1)-6)*5)
    
    
   # meal_data_cleaned['difference_in_glucose_normalized']=(meal_data_cleaned.iloc[:,5:19].max(axis=1)-meal_data_cleaned.iloc[:,22:25].min(axis=1))/(meal_data_cleaned.iloc[:,22:25].min(axis=1))
    #meal_data_cleaned['difference_in_glucose_normalized']=(meal_data_cleaned.iloc[:,7:].max(axis=1)-meal_data_cleaned.iloc[:,6])/(meal_data_cleaned.iloc[:,6])
    difference_in_glucose_normalized=abs((meal_data_cleaned.iloc[:,7:].max(axis=1)-meal_data_cleaned.iloc[:,6])/(meal_data_cleaned.iloc[:,6]))
    
    meal_data_cleaned=meal_data_cleaned.dropna().reset_index().drop(columns='index')
    power_first_max=[]
    index_first_max=[]
    power_second_max=[]
    index_second_max=[]
    for i in range(len(meal_data_cleaned)):
        array=abs(rfft(meal_data_cleaned.iloc[:,0:30].iloc[i].values.tolist())).tolist()
        sorted_array=abs(rfft(meal_data_cleaned.iloc[:,0:30].iloc[i].values.tolist())).tolist()
        sorted_array.sort()
        power_first_max.append(sorted_array[-2])
        power_second_max.append(sorted_array[-3])
        index_first_max.append(array.index(sorted_array[-2]))
        index_second_max.append(array.index(sorted_array[-3]))
    meal_feature_matrix=pd.DataFrame()
    #meal_feature_matrix['tau_time']=meal_data_cleaned['tau_time']
    #meal_feature_matrix['difference_in_glucose_normalized']=meal_data_cleaned['difference_in_glucose_normalized']
    meal_feature_matrix['tau_time']=tau_time
    meal_feature_matrix['difference_in_glucose_normalized']=difference_in_glucose_normalized
    meal_feature_matrix['power_first_max']=power_first_max
    meal_feature_matrix['power_second_max']=power_second_max
    meal_feature_matrix['index_first_max']=index_first_max
    meal_feature_matrix['index_second_max']=index_second_max
    #tm=meal_data_cleaned.iloc[:,22:25].idxmin(axis=1)
    tm = 6
    #maximum=meal_data_cleaned.iloc[:,5:19].idxmax(axis=1)
    maximum=meal_data_cleaned.iloc[:,7:].idxmax(axis=1)
    
    list1=[]
    second_differential_data=[]
    standard_deviation=[]
    for i in range(len(meal_data_cleaned)):
        #list1.append(np.diff(meal_data_cleaned.iloc[:,maximum[i]:tm[i]].iloc[i].tolist()).max())
        list1.append(np.diff(meal_data_cleaned.iloc[i,tm:(maximum[i]+1)].tolist()).max())
        #second_differential_data.append(np.diff(np.diff(meal_data_cleaned.iloc[:,maximum[i]:tm[i]].iloc[i].tolist())).max())
        if(len(meal_data_cleaned.iloc[i,tm:(maximum[i]+1)])>2):
            second_differential_data.append(np.diff(np.diff(meal_data_cleaned.iloc[i,tm:maximum[i]+1].tolist())).max())
        else:
            second_differential_data.append(0)     
        #standard_deviation.append(np.std(meal_data_cleaned.iloc[i]))
    meal_feature_matrix['1stDifferential']=list1
    meal_feature_matrix['2ndDifferential']=second_differential_data
    return meal_feature_matrix


# In[13]:


meal_feature_matrix=createmealfeaturematrix(meal_data)
meal_feature_matrix1=createmealfeaturematrix(meal_data1)
meal_feature_matrix=pd.concat([meal_feature_matrix,meal_feature_matrix1]).reset_index().drop(columns='index')

#print(meal_feature_matrix['difference_in_glucose_normalized'].to_string())


# # Extracting feature matrix for NoMeal Data

# In[14]:


def createnomealfeaturematrix(non_meal_data):
    index_to_remove_non_meal=non_meal_data.isna().sum(axis=1).replace(0,np.nan).dropna().where(lambda x:x>5).dropna().index
    non_meal_data_cleaned=non_meal_data.drop(non_meal_data.index[index_to_remove_non_meal]).reset_index().drop(columns='index')
    non_meal_data_cleaned=non_meal_data_cleaned.interpolate(method='linear',axis=1)
    index_to_drop_again=non_meal_data_cleaned.isna().sum(axis=1).replace(0,np.nan).dropna().index
    non_meal_data_cleaned=non_meal_data_cleaned.drop(non_meal_data_cleaned.index[index_to_drop_again]).reset_index().drop(columns='index')
    non_meal_feature_matrix=pd.DataFrame()
    #non_meal_data_cleaned['tau_time']=(24-non_meal_data_cleaned.iloc[:,0:19].idxmax(axis=1))*5
    #non_meal_data_cleaned['difference_in_glucose_normalized']=(non_meal_data_cleaned.iloc[:,0:19].max(axis=1)-non_meal_data_cleaned.iloc[:,24])/(non_meal_data_cleaned.iloc[:,24])
    tau_time=abs((non_meal_data_cleaned.iloc[:,1:].idxmax(axis=1)-0)*5)
    difference_in_glucose_normalized=abs((non_meal_data_cleaned.iloc[:,1:].max(axis=1)-non_meal_data_cleaned.iloc[:,0])/(non_meal_data_cleaned.iloc[:,0]))
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


# In[15]:


non_meal_feature_matrix=createnomealfeaturematrix(no_meal_data)
non_meal_feature_matrix1=createnomealfeaturematrix(no_meal_data1)
non_meal_feature_matrix=pd.concat([non_meal_feature_matrix,non_meal_feature_matrix1]).reset_index().drop(columns='index')


# # Training the model using DecisionTreeClassifier

# In[16]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


meal_feature_matrix['label']=1
non_meal_feature_matrix['label']=0
total_data=pd.concat([meal_feature_matrix,non_meal_feature_matrix]).reset_index().drop(columns='index')
dataset=shuffle(total_data,random_state=1).reset_index().drop(columns='index')
kfold = KFold(n_splits=10,shuffle=True,random_state=1)
principaldata=dataset.drop(columns='label')
scores_rf = []
accuracy, f1, precision, recall = [], [], [], []
model=DecisionTreeClassifier(criterion="entropy", max_depth = 5)
#model=SVC(C=3000)
for train_index, test_index in kfold.split(principaldata):
    X_train,X_test,y_train,y_test = principaldata.loc[train_index],principaldata.loc[test_index],    dataset.label.loc[train_index],dataset.label.loc[test_index]
    model.fit(X_train,y_train)
    y_test_pred=model.predict(X_test)
    accuracy.append(accuracy_score(y_test,y_test_pred))
    f1.append(f1_score(y_test,y_test_pred))
    precision.append(precision_score(y_test,y_test_pred))
    recall.append(recall_score(y_test,y_test_pred))


# # Calculating and printing different metrics

# In[17]:


#print('Prediction score is',np.mean(scores_rf)*100)
print('Accuracy score is',np.mean(accuracy)*100)
print('F1 Score score is',np.mean(f1)*100)
print('Precision score is',np.mean(precision)*100)
print('Recall score is',np.mean(recall)*100)


# In[18]:


classifier=DecisionTreeClassifier(criterion='entropy')
X, y= principaldata, dataset['label']
classifier.fit(X,y)
dump(classifier, 'DecisionTreeClassifier.pickle')


# In[19]:


# classifier = SVC(kernel='rbf')
# X, y = principaldata, dataset['label']
# classifier.fit(X, y)
# dump(classifier, 'SVC_classifier.joblib')


# In[ ]:




