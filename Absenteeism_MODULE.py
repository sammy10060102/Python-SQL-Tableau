#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Creating a MODULE for later use of the logistic model - getting the ML learning ready for deployment
#storing code in a module will allow us to resue it without trouble


import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator , TransformerMixin

from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[2]:


#CUSTOM SCALER CLASS

class CustomScaler(BaseEstimator,TransformerMixin):
    
    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        self.scaler = StandardScaler(copy,with_mean,with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None
        
    def fit(self,X, y = None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ =np.var(X[self.columns])
        
    def transform(self,X , y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns =self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


# In[3]:


#create a class that we are going to use from here on to predict data

#the class will consist of 5 methods below: init, load and clean data, predicted_probability, predicted_output_catergory,predicted_outputs

class absenteeism_model():
    def __init__(self,model_file,scaler_file):
    #read the 'model' and ' scaler' files which were saved
        with open('model','rb') as model_file, open('scaler','rb') as scaler_file:
            self.reg = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
            self.data = None
            
    
    #take a data file (*.csv) and preprocessed it in the same 
    def load_and_clean_data(self,data_file):
        
        #import the data
        df = pd.read_csv(data_file,delimiter-',')
       
        # store the data in a new variable for later use
        self.df_with_predictions = df.copy()
        
        # drop the "ID" column
        df = df.drop(['ID'], axis=1)
        
        #to preserve the code we have created in the previous section, we will add a column with 'Nan' strings
        df['Absenteeism Time in Hours'] = 'Nan'
        
        #create a seperate dataframe, containing dummy values for ALL available reasons
        reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first = True)
        
        
        #split reason_columns into 4 types
        reason_type_1 = reason_colunms.loc[:,1:14].max(axis=1)
        reason_type_2 = reason_colunms.loc[:,15:17].max(axis=1)
        reason_type_3 = reason_colunms.loc[:,18:21].max(axis=1)
        reason_type_4 = reason_colunms.loc[:,22:].max(axis=1)
        
        #to avoil multicollinearity, drop the "Reason for Absense" column from df
        df =df.drop(['Reason for Absence'],axis =1)
        
        #concatenate df and the 4 types of Reason for Absence
        
        df = pd.concat([df,reason_type_1,reason_type_2, reason_type_3,reason_type_4], axis = 1)
        
        #assign names to the 4 reason type columns
        columns_name = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason 1', 'Reason 2','Reason 3','Reason 4']
        
        df.columns = columns_name
        
        #re-order the columns in df 
        column_names_reordered = ['Reason 1', 'Reason 2','Reason 3','Reason 4','Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education','Children', 'Pets', 'Absenteeism Time in Hours']
        
        #conver the "Date" column into datetime
        df['Date'] = pd.to_datetime(['Date'], format ="%d/%m/%Y")
        
        #creat a list with month values retrieved from the ' Date' column
        list_months =[]
        for i in range(df_reason_mod.shape[0]):
          list_month.append(df_reason_mod['Date'][i].month)
        
        #insert the values in a new column in df, called "Month Value"
        df['Month Value'] = list_months
        
        #create a new feature called "Day of the Week"
        df['Day of the Week'] = df['Date'].appy(lambda x: x.weekday())
                                   
        #drop the "Date" column from df
        df = df.drop(['Date'], axis = 1)
                                   
        #re-order the columns in df
        column_name_upt = ['Reason 1', 'Reason 2','Reason 3','Reason 4','Month Value','Day of the Week','Transportation Expense','Distance to Work', 'Age', 'Daily Work Load Average','Body Mass Index', 'Education','Children', 'Pets', 'Absenteeism Time in Hours']
        df = df[column_name_upt]  
                                   
        # Map "Education" variables : the result is a dummy 
        df['Education'] = df['Education'].map({1:0, 2:1, 3:1,4:1})
        
        #replace the NaN values
        df = df.fillna(value=0)
        
        #drop the original absenteeism time
        df = df.drop(['Absenteeism Time in Hours'],axis=1)
        
        #drop the variables we decide we dont need
        df =df.drop(['Distance to Work','Day of the Week','Daily Work Load Average'], axis=1)
        
        #this line of code if you want to call the "preprocessed data"
        self.preprocessed_data = df.copy()
        
        #we need this line so we can use it in the next functions
        self.data = self.scaler.transform(df)
         


# In[4]:


# a function which outputs the probability of a data point to be 1

def predicted_probability(self):
    if (self.data is not None):
        pred = self.reg.predict(self.data)[:,1]
        return pred
    


# In[5]:


#a funtion which outputs 0 or 1 based on our model

def predicted_output_catergory(self):
    if(self.data is not None):
        pred_outputs = self.reg.predict(self.data)
        return pred_outputs


# In[6]:


#predic outputs and the probabilities and add columns with these values at the end of the new data

def predicted_outputs(self):
    if (self.data is not None):
        self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
        self.preprocessed_date['Prediction']= self.reg.predict(self.data)
        return self.preprocessed_data

