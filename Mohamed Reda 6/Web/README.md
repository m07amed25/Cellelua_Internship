# 🚕 NYC Taxi Fare Predictor

A machine learning web application that predicts NYC taxi fares using an Enhanced XGBoost model with 77.1% accuracy.

## 🎯 Features

- **Interactive Web Interface**: Modern, responsive design with real-time predictions
- **Advanced ML Model**: Enhanced XGBoost with hyperparameter tuning (R² Score: 77.1%, RMSE: $1.65)
- **Comprehensive Feature Engineering**: Distance calculations, time features, weather conditions, and more
- **Real-time Predictions**: Instant fare predictions with detailed breakdown

## 🔧 Technical Stack

- **Backend**: Flask (Python web framework)
- **Machine Learning**: XGBoost, scikit-learn, pandas, numpy
- **Frontend**: HTML5, CSS3, JavaScript (vanilla)
- **Model Persistence**: Joblib

## 📊 Model Performance

- **Model Type**: Enhanced XGBoost Regressor
- **Training Data**: 500,000+ NYC taxi rides
- **R² Score**: 77.11%
- **RMSE**: $1.65
- **Features**: 40+ engineered features including:
  - Geographical coordinates and distances
  - Haversine distance calculations
  - Airport proximity (JFK, LGA, EWR)
  - Time-based features (hour, day, weekday, month)
  - Weather conditions
  - Traffic conditions
  - Car condition

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. Navigate to the Web directory:

```bash
cd Web
```

2. Install required packages:

```bash
pip install flask scikit-learn pandas numpy joblib
```

3. Run the application:

```bash
python app.py
```

4. Open your browser and go to: `http://127.0.0.1:5000`

## 📱 How to Use

1. **Enter Pickup Location**: Provide longitude and latitude for pickup point
2. **Enter Dropoff Location**: Provide longitude and latitude for destination
3. **Set Trip Details**: Choose passenger count and pickup date/time
4. **Select Conditions**: Choose weather, traffic, and car condition
5. **Get Prediction**: Click "Predict Fare" to get instant results

### Sample NYC Coordinates

- **Times Square**: -73.985, 40.758
- **JFK Airport**: -73.778, 40.641
- **Central Park**: -73.965, 40.782
- **Brooklyn Bridge**: -73.996, 40.706

## 🎨 Web Interface Features

- **Modern Design**: Gradient backgrounds and smooth animations
- **Responsive Layout**: Works on desktop, tablet, and mobile devices
- **Real-time Validation**: Form validation with helpful hints
- **Loading States**: Visual feedback during prediction processing
- **Error Handling**: Graceful error messages and recovery

## 📈 API Endpoints

### GET /

Returns the main web interface

### POST /predict

Predicts taxi fare based on input parameters

**Request Body:**

```json
{
  "pickup_longitude": -73.985,
  "pickup_latitude": 40.758,
  "dropoff_longitude": -73.778,
  "dropoff_latitude": 40.641,
  "passenger_count": 1,
  "pickup_datetime": "2024-01-15T14:30",
  "weather": "Clear",
  "traffic_condition": "Light",
  "car_condition": "Good"
}
```

**Response:**

```json
{
  "success": true,
  "predicted_fare": 45.67,
  "distance": 15.3
}
```

## 🔧 Model Features

The prediction model uses 40+ engineered features:

### Core Features

- Pickup/dropoff coordinates
- Passenger count
- Date/time components (hour, day, month, weekday, year)

### Distance Features

- Haversine distance between pickup and dropoff
- Distance to major airports (JFK, LGA, EWR)
- Distance to NYC center
- Distance to Statue of Liberty
- Bearing calculation

### Temporal Features

- Hour sine/cosine encoding
- Weekday dummy variables
- Month dummy variables

### Environmental Features

- Weather condition (Clear, Cloudy, Rainy)
- Traffic condition (Light, Moderate, Heavy)
- Car condition (Bad, Good, Very Good, Excellent)

## 📁 Project Structure

```
Web/
├── app.py                 # Flask application
├── templates/
│   └── index.html        # Web interface
├── static/               # CSS/JS assets (if any)
├── requirements.txt      # Dependencies
└── README.md            # This file

../models/
└── best_taxi_fare_model.pkl  # Trained XGBoost model
```

## 🎯 Development Notes

- The model was trained on historical NYC taxi data with extensive preprocessing
- Features were selected based on correlation analysis and domain knowledge
- Hyperparameter tuning was performed using GridSearchCV
- The web interface provides real-time feedback and error handling

## 🔒 Security & Production Notes

This is a development server. For production deployment:

- Use a production WSGI server (e.g., Gunicorn)
- Implement proper input validation and sanitization
- Add rate limiting and authentication if needed
- Use environment variables for configuration
- Implement proper logging and monitoring

## 📝 License

This project is for educational and demonstration purposes.

---

**Developed with ❤️ using Flask, XGBoost, and modern web technologies**
