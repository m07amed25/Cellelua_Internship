import requests
import json

test_data = {
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

try:
    print("Testing API endpoint...")
    response = requests.post('http://127.0.0.1:5000/predict', json=test_data, timeout=10)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("JSON Response received successfully!")
        print(json.dumps(result, indent=2))
        
        if result.get('success'):
            print(f"Predicted Fare: ${result['predicted_fare']}")
            print(f"Distance: {result['distance']} miles")
            print("JSON serialization error FIXED!")
        else:
            print(f"API Error: {result.get('error')}")
    else:
        print(f"HTTP Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"Connection Error: {e}")
