# F1 Win Probability Predictor

A machine learning project that predicts the probability of an F1 driver winning a race based on various factors including driver skill, car performance, track characteristics, weather conditions, and race strategy.

## Project Overview

This project generates synthetic F1 racing data, trains a machine learning model to predict race outcomes, and provides tools to make predictions for different race scenarios. The model uses features such as:

- Driver skill rating
- Car performance rating  
- Track characteristics (overtaking difficulty, average speed)
- Starting grid position
- Weather conditions
- Tire strategy
- Pit stops

## Files in the Project

- `f1_data_generator.py` - Generates synthetic F1 racing data for training
- `f1_win_predictor.py` - Contains the F1WinPredictor class for training and making predictions
- `f1_prediction_demo.py` - Demonstration script showing how to use the trained model
- `f1_racing_data.csv` - Generated dataset (created by running the data generator)
- `f1_win_predictor.pkl` - Trained model file (created after training)
- `README.md` - This file

## Installation

1. Ensure you have Python 3.7+ installed
2. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn joblib
   ```

## Usage

### 1. Generate Data and Train Model

To generate the dataset and train the model from scratch:

```bash
python f1_win_predictor.py
```

This will:
- Generate 1000 races worth of F1 data (saved to `f1_racing_data.csv`)
- Train a Gradient Boosting classifier to predict win probability
- Evaluate the model performance
- Save the trained model to `f1_win_predictor.pkl`
- Show sample predictions

### 2. Run the Demo

To see example predictions without retraining:

```bash
python f1_prediction_demo.py
```

This loads the pre-trained model and demonstrates:
- Win probabilities for different race scenarios
- Track comparison analysis
- Weather impact analysis
- Grid position impact analysis

### 3. Generate Only Data

To generate a dataset without training:

```bash
python f1_data_generator.py
```

This creates `f1_racing_data.csv` with synthetic F1 racing data.

### 4. Using the Model in Your Own Code

```python
from f1_win_predictor import F1WinPredictor
import pandas as pd

# Load the trained model
predictor = F1WinPredictor()
predictor.load_model('f1_win_predictor.pkl')

# Define a race scenario
driver_data = {
    'driver_name': 'Verstappen',
    'team': 'RedBull',
    'driver_skill': 95,
    'car_performance': 95,
    'track_overtaking_difficulty': 50,  # Medium difficulty track
    'track_avg_speed': 80,              # Average speed in km/h
    'grid_position': 1,                 # Starting from pole
    'weather_encoded': 0,               # 0=dry, 1=mixed, 2=wet
    'tire_encoded': 2,                  # 0=hard, 1=medium, 2=soft, 3=mixed
    'pit_stops': 2                      # Number of planned pit stops
}

# Get win probability
win_prob = predictor.predict_probability(driver_data)
print(f"Win probability: {win_prob:.2%}")

# Predict for multiple drivers in a race
race_entries = [
    {
        'driver_name': 'Verstappen',
        'team': 'RedBull',
        'driver_skill': 95,
        'car_performance': 95,
        'track_overtaking_difficulty': 50,
        'track_avg_speed': 80,
        'grid_position': 1,
        'weather_encoded': 0,
        'tire_encoded': 2,
        'pit_stops': 2
    },
    {
        'driver_name': 'Hamilton',
        'team': 'Mercedes',
        'driver_skill': 94,
        'car_performance': 90,
        'track_overtaking_difficulty': 50,
        'track_avg_speed': 80,
        'grid_position': 3,
        'weather_encoded': 0,
        'tire_encoded': 1,
        'pit_stops': 2
    }
]

results = predictor.predict_race(race_entries)
print(results[['driver', 'win_probability']])
```

## Model Details

The project implements three model types (selectable in `f1_win_predictor.py`):
- **Gradient Boosting** (default) - Ensemble method that builds trees sequentially
- **Random Forest** - Ensemble of decision trees
- **Logistic Regression** - Linear model for probability estimation

Features used for prediction:
1. `driver_skill` - Driver's skill rating (0-100)
2. `car_performance` - Car's performance rating (0-100)
3. `track_overtaking_difficulty` - How difficult it is to overtake on the track (0-100)
4. `track_avg_speed` - Average speed around the track (km/h)
5. `grid_position` - Starting grid position (1-20)
6. `weather_encoded` - Weather condition (0=dry, 1=mixed, 2=wet)
7. `tire_encoded` - Tire strategy (0=hard, 1=medium, 2=soft, 3=mixed)
8. `pit_stops` - Number of planned pit stops

## Data Generation

The synthetic data generator creates realistic F1 racing scenarios with:
- 20 drivers per race with varying skill levels
- Different tracks with unique characteristics
- Variable weather conditions
- Qualifying sessions to determine grid positions
- Race simulations incorporating incidents, safety cars, and tire strategies
- Points allocation based on finishing position (F1 standard)

## Example Output

When running the demo, you'll see outputs like:

```
============================================================
F1 WIN PROBABILITY PREDICTOR - DEMO
============================================================

1. MONACO GP - TOP CONTENDERS
----------------------------------------
   driver_name        team  win_probability
    Verstappen    RedBull           28.3%
      Hamilton   Mercedes           22.1%
      Leclerc    Ferrari           19.7%
      Norris    McLaren           12.4%
      Alonso  AstonMarton           10.5%

2. VERSTAPPEN WIN PROBABILITY BY TRACK (P1 Grid)
----------------------------------------
  Monaco (Street, Hard to Overtake): 22.1%
        Spa (High Speed, Easy Overtake): 38.7%
      Monza (Temple of Speed): 41.2%
  Singapore (Street, Night Race): 26.8%
Silverstone (High Speed Corners): 35.4%

...
```

## Customization

You can modify the following aspects:
- In `f1_data_generator.py`: Adjust number of races, driver/team characteristics, track parameters
- In `f1_win_predictor.py`: Change model type, hyperparameters, feature selection
- In `f1_prediction_demo.py`: Create custom race scenarios for analysis

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- joblib

## License

This project is for educational and demonstration purposes.

## Acknowledgments

- Synthetic data generation based on realistic F1 racing factors
- Machine learning implementation using scikit-learn