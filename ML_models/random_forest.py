from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd 
from sklearn.model_selection import GridSearchCV


# Read dataset
df = pd.read_csv('/content/to_single_row_with_30_instance_per_driver.csv')

# Drop rows with NA values
df = df.dropna(axis = 0)

# Divide dataset into input features, output labels
X, y = df.drop(['Unnamed: 0', 'DrivingStyle', 'DriverID'], axis = 1), df['DrivingStyle']

# encoding categorical values
X = pd.get_dummies(X)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# split dataset into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle = True, random_state=42)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 12)]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(start = 20, stop = 200, num = 12)]

# Method of selecting samples for training each tree
bootstrap = [True]

param_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'bootstrap': bootstrap}

rf_Model = RandomForestClassifier()

rf_Grid = GridSearchCV(estimator = rf_Model, param_grid = param_grid, cv = 5, verbose=2, n_jobs = -1)

rf_Grid.fit(X_train, y_train)

print(rf_Grid.best_params_)

print (f'Train Accuracy - : {rf_Grid.score(X_train,y_train):.3f}')
print (f'Test Accuracy - : {rf_Grid.score(X_test,y_test):.3f}')
