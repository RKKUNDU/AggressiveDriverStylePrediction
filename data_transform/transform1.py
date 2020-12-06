# Create a new dataset with single row for each driver's data
# use max 40 instance of same driver

import pandas as pd

# read dataset
df = pd.read_csv('../input/driverdata/train_data.csv')

# drop unnecessary columns
df = df.drop(['DateTimeOfTrip', 'IDOfPrecedingVehicle'], axis = 1)

# drop rows with missing values
df = df.dropna(axis = 0)

# create groups by DriverID
groups = df.groupby(['DriverID'])

# create a new dataframe for storing the modified dataset
new_df = pd.DataFrame()
index = 0

# these numeric features will be used
features = ["Lane", 'Speed', 'SpeedOfPrecedingVehicle', 'WeightOfPrecedingVehicle', 'LengthOfPrecedingVehicle', 'TimeGapWithPrecedingVehicle', 'RoadCondition', 'AirTemperature', 'PrecipitationType', 'PrecipitationIntensity', 'RelativeHumidity', 'WindDirection', 'WindSpeed', 'LightingCondition']

instances = 40

# iterate over all groups
for group in groups:
    # store common info of a driver
    new_df.loc[index, f'DriverID'] = group[0]
    new_df.loc[index, f'LengthOfVehicle'] = group[1].iloc[0][f'LengthOfVehicle']
    new_df.loc[index, f'Weight'] = group[1].iloc[0][f'Weight']
    new_df.loc[index, f'NumberOfAxles'] = group[1].iloc[0][f'NumberOfAxles']
    
    # find mean & max of various features. It will be used in place of missing values like Lane29 
    mean_ = group[1].mean()
    max_ = group[1].max()
    
    i = 0
    # convert multiple rows of driver's data into a single row
    for i in range(group[1].shape[0]):        
        for feature in features:
            new_df.loc[index, f'{feature}{i}'] = group[1].iloc[i][f'{feature}']
            
        # if there are more than `instances` datapoint, we ignore rest
        if i >= instances:
            break
            
    # if there are less than `instances` datapoints for a driver, fill with previously found mean/max
    while i <= instances:
        for feature in features:
            # categorical features -> use max
            if feature in ['RoadCondition', 'PrecipitationType', 'PrecipitationIntensity', 'LightingCondition']:
                new_df.loc[index, f'{feature}{i}'] = max_[f'{feature}']
            # numerical features -> use mean
            else:
                new_df.loc[index, f'{feature}{i}'] = mean_[f'{feature}']
                
        i += 1
    
    new_df.loc[index, f'DrivingStyle'] = group[1].iloc[0][f'DrivingStyle']
    index += 1
    
# for col_name in new_df.columns:
#     print(col_name)
    
    
# verify whether it contains any missing values
new_df.isna().any(axis = None)
