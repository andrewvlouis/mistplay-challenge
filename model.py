########################################################################################################

# IMPORTS

import pandas as pd
import requests
import json
from imblearn.under_sampling import RandomUnderSampler 
import os
from xgboost import XGBClassifier
import pickle
import re

########################################################################################################

# PREPROCESSING

df = pd.read_csv('DataScienceChallenge.csv').drop(['Unnamed: 27', 'Unnamed: 28'], axis = 1) #read in data
df['y'] = df['y'].apply(lambda x: 1 if x > 0 else 0) # make binary

# took out x1 because it looked like an id. not necessary for prediction
# took out x3 because x15 had enough info. the entire model of the phone does not seem necessary
# took out x24 because it was extremely sparse did not improve the predictive power of the model
# took out x25 for the same reason

X = df[['x2' , 'x4','x5','x6', 'x7','x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', \
       'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21', 'x22', 'x23', 'x26']] 
y = df[['y']]
X['x2'] = X['x2'].astype(str) #make version a string
X['x26'] = X['x26'].fillna('Unknown') # fill na with unknown
X['x26'] = X['x26'].apply(lambda x: re.sub('[^A-Za-z0-9]+', '', x)) #remove spaces and funky chars from string
X = pd.get_dummies(X)
X = X.as_matrix()

########################################################################################################

# UNDERSAMPLING

# If we were to pass this through a naive model, i.e. an untuned tree or baye's classifier, we may get a decent accuracy. However, the ability of the model to predict the negative class would be poor.

# What we wish to do is reduce the bias of the model towards the 0 lass. We do this by using an undersampling technique.

# Below, we use random under sampling, which randomly selected datapoints from the majority class and deletes them until the classes are balanced.

# Random under sampling was chosen due to the fact that the data set provided was small. If we chose oversampling, (i.e. SMOTE, random over sample, etc.), overfitting would be bound to happen. 


sm = RandomUnderSampler() # undersample
X, y = sm.fit_sample(X, y) # fit undersample method

########################################################################################################

# MODEL FIT


os.environ['KMP_DUPLICATE_LIB_OK']='True' #xg boost produces an error frequently on mac osx. this line is a workaround to avoid it

model = XGBClassifier(learning_rate= 0.01, # for performance
                        n_estimators= 100, # because of onehot encoding
                        max_depth= 3, #XG Boost standard
                        subsample= 0.8, #XG Boost standard
                        colsample_bytree= 0.55, 
                        gamma= 1) #XG Boost standard
model.fit(X, y) # fit model 

########################################################################################################

# PICKLE MODEL

pickle.dump(model, open('model.pkl','wb'))# dump model