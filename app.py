from flask import Flask, render_template, request, jsonify
from forecast import get_weather_data, predict_fire_indices, calculate_fwi, check_fire_risk
from manual import predict as manual_predict
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/manual', methods=['POST'])
def get_manual_prediction():
    try:
        data = request.json
        print("Received JSON:", data)  # Debugging input data

        if not data:
            return jsonify({'error': 'No data received'})

        result = manual_predict(data)
        print("Prediction result:", result)  # Debugging model output

        return jsonify(result)
    except Exception as e:
        print("Error occurred:", str(e))  # Debugging error
        return jsonify({'error': str(e)})
    
    
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    location = data["location"] 
    start_date = data["startDate"]
    end_date = data["endDate"]
    
    weather_data = get_weather_data(location, start_date, end_date, api_key="9ZG928AFMYKNSXDJ4MZAP8HCS")
    
    if weather_data:
        results = []
        for day in weather_data:
            temperature = day["temperature"]
            rh = day["relative_humidity"]
            ws = day["wind_speed"]
            rain = day["rainfall"]

            ffmc, dmc, dc, isi = predict_fire_indices(temperature, rh, ws, rain)
            row = {"FFMC": ffmc, "DMC": dmc, "DC": dc, "ISI": isi}
            bui, fwi = calculate_fwi(row)
            fire_risk = check_fire_risk(fwi)

            results.append({
                "date": day["date"],
                "temperature": temperature,
                "relative_humidity": rh,
                "wind_speed": ws,
                "rainfall": rain,
                "FFMC": ffmc,
                "DMC": dmc,
                "DC": dc,
                "ISI": isi,
                "BUILD_UP_INDEX": bui,
                "FIRE_WEATHER_INDEX": fwi,
                "FIRE_RISK": fire_risk
            })

        return jsonify(results)
    
    return jsonify({"error": "Failed to fetch weather data"})

if __name__ == '__main__':
    app.run(debug=True)

