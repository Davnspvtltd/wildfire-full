import math
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('forest_fire.csv')
data = data.drop('Unnamed: 0', axis=1)

imputer = SimpleImputer(strategy='mean')
X = data.drop(columns=['Classes'])  
y = data['Classes']  

X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

def calculate_fwi(row):
    ffmc = row['FFMC'] 
    dmc = row['DMC']    
    dc = row['DC']     
    isi = row['ISI']    

    if dmc > 0:
        bui = 0.8 * dmc * (dc / (dmc + 0.4 * dc))
    else:
        bui = 0
    
    if bui > 0:
        fwi = math.exp(2.72 * (0.434 * math.log(bui) + 0.300)) * isi
    else:
        fwi = 0
    
    return pd.Series([bui, fwi])

def predict(data):
    user_input = pd.DataFrame([[data['temp'], data['rh'], data['wind'], data['rain'], data['ffmc'], data['dmc'], data['dc'], data['isi']]], 
                               columns=['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI'])

    user_input[['BUILD_UP_INDEX', 'fireweather_index']] = user_input.apply(calculate_fwi, axis=1)
    user_input_imputed = imputer.transform(user_input)
    prediction = rf.predict(user_input_imputed)

    fireweather_index = user_input['fireweather_index'].values[0]

    min_fwi = 0
    max_fwi = 30656.085019 

    fireweather_percentage = ((fireweather_index - min_fwi) / (max_fwi - min_fwi)) * 100
    fireweather_percentage = min(max(fireweather_percentage, 0), 100)

    response = {
        'prediction': 'There is a fire risk.' if prediction[0] == 1 else 'No fire risk.',
        'fireweather_index': fireweather_index,
        'fireweather_percentage': fireweather_percentage
    }

    return response
