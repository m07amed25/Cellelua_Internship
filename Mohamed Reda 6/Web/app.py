from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

sys.path.append('..')

app = Flask(__name__)

model_path = r'C:\Users\CRIZMA\Desktop\Cellelua\Mohamed Reda 6\models\best_taxi_fare_model.pkl'
model = joblib.load(model_path)

def create_features(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude,
                   passenger_count, pickup_datetime, weather, traffic_condition, car_condition):
    
    def haversine_distance(lon1, lat1, lon2, lat2):
        R = 3959 
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    distance = haversine_distance(pickup_longitude, pickup_latitude, 
                                 dropoff_longitude, dropoff_latitude)
    
    lon1, lat1, lon2, lat2 = map(np.radians, [pickup_longitude, pickup_latitude,
                                             dropoff_longitude, dropoff_latitude])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.cos(lat2) * np.cos(dlon) - np.sin(lat1) * np.sin(lat2)
    bearing = np.arctan2(y, x)
    bearing = np.degrees(bearing)
    bearing = (bearing + 360) % 360
    
    jfk_dist = haversine_distance(pickup_longitude, pickup_latitude, -73.7781, 40.6413)
    lga_dist = haversine_distance(pickup_longitude, pickup_latitude, -73.8740, 40.7769)
    ewr_dist = haversine_distance(pickup_longitude, pickup_latitude, -74.1745, 40.6895)
    
    nyc_dist = haversine_distance(pickup_longitude, pickup_latitude, -73.9857, 40.7484)
    
    sol_dist = haversine_distance(pickup_longitude, pickup_latitude, -74.0445, 40.6892)
    
    dt = pd.to_datetime(pickup_datetime)
    hour = dt.hour
    day = dt.day
    month = dt.month
    year = dt.year
    weekday = dt.weekday()
    day_of_year = dt.timetuple().tm_yday
    
    features = {}
    
    features['passenger_count'] = passenger_count
    features['hour'] = hour
    features['day'] = day
    features['year'] = year
    features['day_of_year'] = day_of_year
    
    features['jfk_dist'] = jfk_dist
    features['ewr_dist'] = ewr_dist
    features['lga_dist'] = lga_dist
    features['sol_dist'] = sol_dist
    features['nyc_dist'] = nyc_dist
    features['distance'] = distance
    features['bearing'] = bearing
    
    features['distance_log'] = np.log1p(distance)
    features['distance_squared'] = distance ** 2
    features['distance_x_hour'] = distance * hour
    
    features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    
    car_mapping = {'Bad': 0, 'Good': 1, 'Very Good': 2, 'Excellent': 3}
    features['Car Condition'] = car_mapping.get(car_condition, 1)
    
    traffic_mapping = {'Light': 0, 'Moderate': 1, 'Heavy': 2}
    features['Traffic Condition'] = traffic_mapping.get(traffic_condition, 1)
    
    weather_mapping = {
        'Clear': 'sunny',
        'Cloudy': 'stormy', 
        'Rainy': 'rainy',
        'Windy': 'windy'
    }
    weather_encoded = weather_mapping.get(weather, 'sunny')
    
    features['Weather_rainy'] = 0
    features['Weather_stormy'] = 0  
    features['Weather_sunny'] = 0
    features['Weather_windy'] = 0
    
    features[f'Weather_{weather_encoded}'] = 1
    
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    for day_name in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Saturday', 'Sunday']:
        features[f'weekday_{day_name}'] = 0
    
    if weekday < 4:  # Monday-Thursday
        features[f'weekday_{weekday_names[weekday]}'] = 1
    elif weekday == 4:  # Friday - map to Thursday or Saturday
        features['weekday_Thursday'] = 1  # Default Friday to Thursday
    elif weekday >= 5:  # Saturday, Sunday
        features[f'weekday_{weekday_names[weekday]}'] = 1
    
    month_names = ['January', 'February', 'March', 'May', 'June', 'July', 
                   'August', 'September', 'October', 'November', 'December']
    
    for month_name in month_names:
        features[f'month_{month_name}'] = 0
    
    month_mapping = {
        1: 'January', 2: 'February', 3: 'March', 4: 'March',  # April maps to March
        5: 'May', 6: 'June', 7: 'July', 8: 'August', 
        9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }
    
    month_name = month_mapping.get(month, 'January')
    features[f'month_{month_name}'] = 1
    
    return features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        pickup_longitude = float(data['pickup_longitude'])
        pickup_latitude = float(data['pickup_latitude'])
        dropoff_longitude = float(data['dropoff_longitude'])
        dropoff_latitude = float(data['dropoff_latitude'])
        passenger_count = int(data['passenger_count'])
        pickup_datetime = data['pickup_datetime']
        weather = data['weather']
        traffic_condition = data['traffic_condition']
        car_condition = data['car_condition']
        
        features = create_features(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude,
                                 passenger_count, pickup_datetime, weather, traffic_condition, car_condition)
        
        expected_features = [
            'Car Condition', 'Traffic Condition', 'passenger_count', 'hour', 'day', 'year',
            'jfk_dist', 'ewr_dist', 'lga_dist', 'sol_dist', 'nyc_dist', 'distance', 'bearing',
            'Weather_rainy', 'Weather_stormy', 'Weather_sunny', 'Weather_windy',
            'month_August', 'month_December', 'month_February', 'month_January', 'month_July',
            'month_June', 'month_March', 'month_May', 'month_November', 'month_October',
            'month_September', 'weekday_Monday', 'weekday_Saturday', 'weekday_Sunday',
            'weekday_Thursday', 'weekday_Tuesday', 'weekday_Wednesday', 'distance_x_hour',
            'distance_squared', 'distance_log', 'hour_sin', 'hour_cos', 'day_of_year'
        ]
        
        feature_values = []
        for feature_name in expected_features:
            feature_values.append(features.get(feature_name, 0))
        
        X = pd.DataFrame([feature_values], columns=expected_features)
        
        prediction = model.predict(X)[0]
        
        predicted_fare = float(prediction)
        distance_value = float(features['distance'])
        
        return jsonify({
            'success': True,
            'predicted_fare': round(predicted_fare, 2),
            'distance': round(distance_value, 2)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
