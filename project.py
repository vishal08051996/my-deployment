# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 18:41:42 2022

@author: LENOVO
"""



## or use my computor to load the data set.
import pandas as pd
import numpy as np
data = pd.read_csv("G:/data science/360PROJECT_2/venky_duplicate/Project.csv",encoding='latin1')

import numpy as np ## numerical(mathematical) calculations

## Understanding the data
data.info()  ## information about the null,data type, memory
data.describe() ## statistical information
data.shape ## (525, 18)
data.columns

'''
(['S.No', 'Train_Names', 'company_name', 'country_name', 'engine_type',
       'Train_Type', 'Train_bogies', 'speed_of_train', 'source_station',
       'destination_station', 'carry_weight', 'maintance_cost', 'Hours',
       'Price'],
 
 '''
 
"""['s.no', 'Institute', 'Subject', 'Location', 'Trainer_Qualification',
       'Online_classes', 'Offline_classes', 'Trainer_experiance',
       'Course_level', 'Course_hours', 'Course_rating', 'Rental_permises',
       'Trainer_slary', 'Maintaince_cost', 'Non_teaching_staff_salary',
       'Placements', 'Certificate', 'Price']"""

####
data.drop(['s.no'], axis = 1, inplace = True)
data.shape # (525,17)
####


data=data.iloc[:,4:]

## Data cleaning

data.duplicated().sum() ## no duplicates
data.isna().sum() # no null values

## Label encoder (Converting categorical into numeric)
cols = [ 'engine_type',
       'Train_Type', 'Train_bogies', 'speed_of_train', 'source_station',
       'destination_station', 'carry_weight', 'maintance_cost', 'Hours']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Instantiate the encoders
encoders = {column: le for column in cols}

for column in cols:
    data[column] = encoders[column].fit_transform(data[column])

data.info()

## check ing outliers
import seaborn as sns
import matplotlib.pyplot as plt

for i in cols:
    sns.boxplot(data[i]); plt.show() 


data.var()



########
## Certificate column has zero variance, so drop it
data.drop(['Certificate'], axis = 1, inplace = True)
###########



import statsmodels.formula.api as smf 
         
ml1 = smf.ols('Price ~ engine_type + Train_Type + Train_bogies + speed_of_train + source_station + destination_station + carry_weight + maintance_cost + Hours', data = data).fit() # regression model

# Summary
ml1.summary()

from statsmodels.stats.outliers_influence import variance_inflation_factor

X = data[list(data.select_dtypes(include = ['int64', 'float64']).columns)]

# Profit feature is dependent or out put feature so we are deleting
X = X.drop('Price', axis = 1)

## VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

## calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(len(X.columns))]
print(vif_data)

# Split the data set into train(80% of the data) and test(20% of the data)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data.drop("Price", axis = 1), data.Price, test_size = 0.2, random_state = 42)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, make_scorer
rmse =lambda y, y_hat: np.sqrt(mean_squared_error(y, y_hat))

lm = LinearRegression()
lm.fit(x_train, y_train)

lm_preds = lm.predict(x_test)
rmse(y_test, lm_preds)

## Applied TOPT Regressor and got the best best model.

from tpot import TPOTRegressor
rmse_scorer = make_scorer(rmse, greater_is_better = False)
pipeline_optimizer = TPOTRegressor(
    scoring = rmse_scorer,
    max_time_mins = 2,
    random_state = 42,
    verbosity = 2
    )
pipeline_optimizer.fit(x_train, y_train)

print(pipeline_optimizer.score(x_test, y_test))

pipeline_optimizer.fitted_pipeline_
pipeline_optimizer.export('Mock_p2.py')

#############my


from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from tpot.export_utils import set_param_recursive


# Average CV score on the training set was: -363.275186717333
exported_pipeline = make_pipeline(
    MaxAbsScaler(),
    LassoLarsCV(normalize=False)
)


exported_pipeline.fit(x_train, y_train)

y_pred_test = exported_pipeline.predict(x_test)

result_test = pd.DataFrame({'Actual':y_test, "Predicted": y_pred_test})
result_test.head(10)

############################################
## Best fit model venky

from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

exported_pipeline = make_pipeline(
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_features=0.05, min_samples_leaf=19, min_samples_split=17, n_estimators=100)),
    StackingEstimator(estimator=AdaBoostRegressor(learning_rate=1.0, loss="linear", n_estimators=100)),
    StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.9000000000000001, min_samples_leaf=3, min_samples_split=5, n_estimators=100)),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    ExtraTreesRegressor(bootstrap=False, max_features=0.6500000000000001, min_samples_leaf=1, min_samples_split=4, n_estimators=100)
)


exported_pipeline.fit(x_train, y_train)

y_pred_test = exported_pipeline.predict(x_test)

result_test = pd.DataFrame({'Actual':y_test, "Predicted": y_pred_test})
result_test.head(10)
###############################################################







## importing r2_score module
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# predicting the accuracy score
score_test = r2_score(y_test, y_pred_test)

print('R2 score(test): ', score_test)
print('Mean squared error(test): ', mean_squared_error(y_test, y_pred_test))
print('Root Mean squared error(test): ', np.sqrt(mean_squared_error(y_test, y_pred_test)))

""" 
R2 score(test):  0.9846734738249692
Mean squared error(test):  1728.420359021164
Root Mean squared error(test):  41.574275207406366
"""

y_pred_train = exported_pipeline.predict(x_train)

result_train = pd.DataFrame({'Actual':y_train, "Predicted": y_pred_train})
result_train.head(10)

score_train = r2_score(y_train, y_pred_train)

print('R2 score(train): ', score_train)
print('Mean squared error(train): ', mean_squared_error(y_train, y_pred_train))
print('Root Mean squared error(train): ', np.sqrt(mean_squared_error(y_train, y_pred_train)))

"""
R2 score(train):  0.9999877586156279
Mean squared error(train):  1.4589904497354576
Root Mean squared error(train):  1.2078867702460598
"""
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred_test)


import pickle

pickle.dump(exported_pipeline, open('data_model_p2.pkl', 'wb'))

# Load the model from disk
model = pickle.load(open("data_model_p2.pkl", "rb"))

result = model.score(x_test, y_test)
print(result)

#Model predict the course price correctly based on features(15).

import pickle
data.to_csv("data_p2.csv", index=False)
pickle.dump(exported_pipeline, open('data_end_p2.pkl','wb'))


