# Tutorial from https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
# features: variables
# labels: y value

# Pandas is used for data manipulation
import pandas as pd
# Use numpy to convert to arrays
import numpy as np
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Read in data and display first 5 rows
features = pd.read_csv('temps.csv')
print(features.head(5))

print('The shape of our features is: i.e. (row X cols)', features.shape)

# This process tries to remote the anomanies
# Descriptive statistics for each column
print(features.describe())
#Another way to verify the quality of the data is to make basic plots

# Data preparation
# panda one-hot encode the data using the pandas get_dummies
features = pd.get_dummies(features)
# Display the first 5 rows of the last 12 columns
# i.e. all rows and column starts from the 5th
print(features.iloc[:,5:].head(5))

print('===========	Grab the y data and data for factors	============')
# Labels are the values we want to predict (i.e. y value)
# this returns an array of the 'actual' column
labels = np.array(features['actual'])

# Remove the labels from the features
# axis 1 refers to the columns
# we drop the column as the actual column is the y value while features means all data for the factors
features= features.drop('actual', axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)

# Convert to numpy array
features = np.array(features)

print('===========	Separate training and testing sets	============')
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
print(test_features, test_labels)

print('===========	End of preparing data	============')




