import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import math

def get_weather_data(location, start_date, end_date, api_key):
    """
    Fetches weather data, calculates relative humidity, and returns 
    temperature, rainfall, relative humidity, and wind speed.
    
    Parameters:
        location (str): City name (e.g., "London,UK").
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        api_key (str): API key for weather data.
    
    Returns:
        list of dict: Weather data containing temp, rainfall, RH, and wind speed.
    """
    
    # API URL
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{start_date}/{end_date}?key={api_key}"
    
    try:
        # Send request to API
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes

        # Parse JSON response
        data = response.json()
        results = []

        # Define function to calculate RH using Magnus-Tetens formula
        def calculate_relative_humidity(temp_c, dew_point_c):
            A, B = 17.625, 243.04
            svp = np.exp((A * temp_c) / (B + temp_c))  # Saturation vapor pressure
            avp = np.exp((A * dew_point_c) / (B + dew_point_c))  # Actual vapor pressure
            return round((avp / svp) * 100, 2)

        # Loop through days in response
        for day in data.get("days", []):
            temp_c = (day.get("temp") - 32) * 5/9  # Convert to Celsius
            dew_point_c = (day.get("dew") - 32) * 5/9  # Convert to Celsius
            rainfall = day.get("precip", 0.0)  # Rainfall in inches
            wind_speed = day.get("windspeed", 0.0)  # Wind Speed in mph

            # Compute Relative Humidity
            relative_humidity = calculate_relative_humidity(temp_c, dew_point_c)

            # Append results
            results.append({
                "date": day.get("datetime"),
                "temperature": round(temp_c, 2),
                "rainfall": rainfall,
                "relative_humidity": relative_humidity,
                "wind_speed": wind_speed
            })

        return results  # Return list of dictionaries

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

def predict_fire_indices(temperature, rh, ws, rain, data_path="forest_fire.csv"):
    # Load dataset
    data = pd.read_csv(data_path)
    
    # Define features and target variables
    X = data[['Temperature', 'RH', 'Ws', 'Rain']]
    y = data[['FFMC', 'DMC', 'DC', 'ISI']]
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Transform input data
    input_data = np.array([[temperature, rh, ws, rain]])
    input_data_scaled = scaler.transform(input_data)
    
    # Predict fire indices
    prediction = model.predict(input_data_scaled)
    return prediction[0]  # Return predicted FFMC, DMC, DC, ISI

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

def check_fire_risk(fwi):
    """
    Checks if the Fire Weather Index (FWI) indicates a fire risk.
    
    Parameters:
        fwi (float): Fire Weather Index value.
    
    Returns:
        str: "Risk" if FWI > 500, else "No Risk".
    """
    if fwi > 500:
        return "Risk"
    else:
        return "No Risk"

# Example Usage
api_key = "9ZG928AFMYKNSXDJ4MZAP8HCS"
location = "Tulare County, California"
start_date = "2024-08-03"
end_date = "2024-08-04"

# Fetch weather data
weather_data = get_weather_data(location, start_date, end_date, api_key)

if weather_data:
    for day in weather_data:
        temperature = day["temperature"]
        rh = day["relative_humidity"]
        ws = day["wind_speed"]
        rain = day["rainfall"]

        # Predict fire indices
        ffmc, dmc, dc, isi = predict_fire_indices(temperature, rh, ws, rain)

        # Calculate FWI and BUI
        row = pd.Series({'FFMC': ffmc, 'DMC': dmc, 'DC': dc, 'ISI': isi})
        bui, fwi = calculate_fwi(row)

        # Check fire risk
        fire_risk = check_fire_risk(fwi)

        # Output results
        print(f"Date: {day['date']}")
        print(f"Temperature: {temperature}°C")
        print(f"Relative Humidity: {rh}%")
        print(f"Wind Speed: {ws} mph")
        print(f"Rainfall: {rain} inches")
        print(f"FFMC: {ffmc}")
        print(f"DMC: {dmc}")
        print(f"DC: {dc}")
        print(f"ISI: {isi}")
        print(f"BUILD_UP_INDEX: {bui}")
        print(f"FIRE_WEATHER_INDEX: {fwi}")
        print(f"FIRE RISK: {fire_risk}")
        print("-" * 40)