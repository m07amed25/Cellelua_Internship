#!/usr/bin/env python3

import sys
import os
sys.path.append('..')

try:
    # Test model loading
    print("🔍 Testing model loading...")
    import joblib
    model_path = '../models/best_taxi_fare_model.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"✅ Model loaded successfully: {type(model).__name__}")
    else:
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)
    
    # Test feature creation
    print("\n🔍 Testing feature engineering...")
    from app import create_features
    
    sample_features = create_features(
        pickup_longitude=-73.985,
        pickup_latitude=40.758, 
        dropoff_longitude=-73.778,
        dropoff_latitude=40.641,
        passenger_count=1,
        pickup_datetime='2024-01-15T14:30',
        weather='Clear',
        traffic_condition='Light',
        car_condition='Good'
    )
    print(f"✅ Features created: {len(sample_features)} features")
    
    # Test prediction
    print("\n🔍 Testing prediction...")
    import pandas as pd
    import numpy as np
    
    # Expected feature order from training
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
    
    # Create feature vector
    feature_values = [sample_features.get(f, 0) for f in expected_features]
    X = pd.DataFrame([feature_values], columns=expected_features)
    
    prediction = model.predict(X)[0]
    print(f"✅ Prediction successful: ${prediction:.2f}")
    
    # Test API endpoint
    print("\n🔍 Testing API endpoint...")
    import requests
    import json
    
    try:
        data = {
            'pickup_longitude': -73.985,
            'pickup_latitude': 40.758, 
            'dropoff_longitude': -73.778,
            'dropoff_latitude': 40.641,
            'passenger_count': 1,
            'pickup_datetime': '2024-01-15T14:30',
            'weather': 'Clear',
            'traffic_condition': 'Light',
            'car_condition': 'Good'
        }
        
        response = requests.post('http://127.0.0.1:5000/predict', json=data, timeout=5)
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"✅ API test successful: ${result['predicted_fare']}")
            else:
                print(f"❌ API returned error: {result.get('error')}")
        else:
            print(f"❌ API request failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ API connection failed: {e}")
    
    print("\n🎉 All tests completed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
