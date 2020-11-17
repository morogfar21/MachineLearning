# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

#%% a plot disease progression

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X_age = diabetes_X[:, np.newaxis, 0]
diabetes_X_sex = diabetes_X[:, np.newaxis, 1]
diabetes_X_bmi = diabetes_X[:, np.newaxis, 2]
diabetes_X_bp = diabetes_X[:, np.newaxis, 3]

# Plot outputs
plt.scatter(diabetes_X_age, diabetes_y,  color='black')
plt.title("Diabetes")
plt.xlabel("Age")
plt.ylabel("Disease Progression")
plt.xticks(())
plt.yticks(())
plt.show()

plt.scatter(diabetes_X_sex, diabetes_y,  color='black')
plt.title("Diabetes")
plt.xlabel("Sex")
plt.ylabel("Disease Progression")
plt.xticks(())
plt.yticks(())
plt.show()

plt.scatter(diabetes_X_bmi, diabetes_y,  color='black')
plt.title("Diabetes")
plt.xlabel("BMI")
plt.ylabel("Disease Progression")
plt.xticks(())
plt.yticks(())
plt.show()

plt.scatter(diabetes_X_bp, diabetes_y,  color='black')
plt.title("Diabetes")
plt.xlabel("Blood Pressure")
plt.ylabel("Disease Progression")
plt.xticks(())
plt.yticks(())
plt.show()

#%% b predict disease progression from 1 feature

# Split the data into training/testing sets
diabetes_X_bmi_train = diabetes_X_bmi[:-20]
diabetes_X_bmi_test = diabetes_X_bmi[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_bmi_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_bmi_test)

#%% c plot data and prediction line
# Plot outputs
plt.scatter(diabetes_X_bmi_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_bmi_test, diabetes_y_pred, color='blue', linewidth=3)
plt.title("Diabetes - Predicted")
plt.xlabel("BMI")
plt.ylabel("Disease Progression")
plt.xticks(())
plt.yticks(())
plt.show()


#%% d plot residuals

# The coefficients
print('Coefficients: ', regr.coef_)

# The mean squared error
mse = mean_squared_error(diabetes_y_test, diabetes_y_pred)
print('Mean squared error: %.2f' % mse )

# The root mean squared error
rmse = np.sqrt(mse)
print('Root mean squared error: %.2f' % rmse)

# The coefficient of determination: 1 is perfect prediction
r2score = r2_score(diabetes_y_test, diabetes_y_pred)
print('Coefficient of determination: %.2f' % r2score)

from yellowbrick.regressor import residuals_plot
viz = residuals_plot(regr, diabetes_X_bmi_train, diabetes_y_train, diabetes_X_bmi_test, diabetes_y_test)

# de er ikke rigtig normalfordelte.
# der er flere outliers, som er med til at 

#%% e predict disease progression from first 4 features

# use first 4 features
diabetes_X_4 = diabetes_X[:, 0:4]
# print(diabetes_X_4)
# print(diabetes_X_4.T[0])

# # Plot outputs
# plt.scatter(diabetes_X_4.T[0], diabetes_y,  color='red')
# plt.scatter(diabetes_X_4.T[1], diabetes_y,  color='blue')
# plt.scatter(diabetes_X_4.T[2], diabetes_y,  color='green')
# plt.scatter(diabetes_X_4.T[3], diabetes_y,  color='black')
# plt.title("Diabetes")
# plt.xlabel("First 4 features")
# plt.ylabel("Disease Progression")
# plt.xticks(())
# plt.yticks(())
# plt.show()

# Split the data into training/testing sets
diabetes_X_4_train = diabetes_X_4[:-20]
diabetes_X_4_test = diabetes_X_4[-20:]

# Split the targets into training/testing sets
diabetes_y_4_train = diabetes_y[:-20]
diabetes_y_4_test = diabetes_y[-20:]

# Create linear regression object
regr4 = linear_model.LinearRegression()

# Train the model using the training sets
regr4.fit(diabetes_X_4_train, diabetes_y_4_train)

# Make predictions using the testing set
diabetes_y_4_pred = regr4.predict(diabetes_X_4_test)

# The mean squared error
mse4 = mean_squared_error(diabetes_y_4_test, diabetes_y_4_pred)
print('Mean squared error: %.2f' % mse4 )

# The root mean squared error
rmse4 = np.sqrt(mse4)
print('Root mean squared error: %.2f' % rmse4)

# The coefficient of determination: 1 is perfect prediction
r2score4 = r2_score(diabetes_y_4_test, diabetes_y_4_pred)
print('Coefficient of determination: %.2f' % r2score4)


#%% f predict disease progression from all features

# Split the data into training/testing sets
diabetes_X_train_all = diabetes_X[:-20]
diabetes_X_test_all = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create linear regression object
regr_all = linear_model.LinearRegression()

# Train the model using the training sets
regr_all.fit(diabetes_X_train_all, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred_all = regr_all.predict(diabetes_X_test_all)

# The mean squared error
mse_all = mean_squared_error(diabetes_y_test, diabetes_y_pred_all)
print('Mean squared error: %.2f' % mse_all )

# The root mean squared error
rmse_all = np.sqrt(mse_all)
print('Root mean squared error: %.2f' % rmse_all)

# The coefficient of determination: 1 is perfect prediction
r2score_all = r2_score(diabetes_y_test, diabetes_y_pred_all)
print('Coefficient of determination: %.2f' % r2score_all)

#%% g train/test split 70/30

diabetes_size = len(diabetes_X)
print("Data set size:", diabetes_size)

test_size = int(diabetes_size*0.3)
print("Test set size:", test_size)

# Split the data into training/testing sets
diabetes_X_train_all_7030 = diabetes_X[:-test_size]
diabetes_X_test_all_7030 = diabetes_X[-test_size:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-test_size]
diabetes_y_test = diabetes_y[-test_size:]

# Create linear regression object
regr_all_7030 = linear_model.LinearRegression()

# Train the model using the training sets
regr_all_7030.fit(diabetes_X_train_all_7030, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred_all_7030 = regr_all.predict(diabetes_X_test_all_7030)

# The mean squared error
mse_all_7030 = mean_squared_error(diabetes_y_test, diabetes_y_pred_all_7030)
print('Mean squared error: %.2f' % mse_all_7030 )

# The root mean squared error
rmse_all_7030 = np.sqrt(mse_all_7030)
print('Root mean squared error: %.2f' % rmse_all_7030)

# The coefficient of determination: 1 is perfect prediction
r2score_all_7030 = r2_score(diabetes_y_test, diabetes_y_pred_all_7030)
print('Coefficient of determination: %.2f' % r2score_all_7030)
