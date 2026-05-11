"""
F1 Prediction Demo - Quick usage examples
"""
from f1_win_predictor import F1WinPredictor
import pandas as pd


def main():
    # Load trained model
    predictor = F1WinPredictor()
    predictor.load_model('f1_win_predictor.pkl')

    print("=" * 60)
    print("F1 WIN PROBABILITY PREDICTOR - DEMO")
    print("=" * 60)

    # Example 1: Predict a specific race scenario
    print("\n1. MONACO GP - TOP CONTENDERS")
    print("-" * 40)

    monaco_race = [
        {'driver_name': 'Verstappen', 'team': 'RedBull',
         'driver_skill': 95, 'car_performance': 95,
         'track_overtaking_difficulty': 95, 'track_avg_speed': 60,
         'grid_position': 1, 'weather_encoded': 0,
         'tire_encoded': 2, 'pit_stops': 2},

        {'driver_name': 'Hamilton', 'team': 'Mercedes',
         'driver_skill': 94, 'car_performance': 90,
         'track_overtaking_difficulty': 95, 'track_avg_speed': 60,
         'grid_position': 2, 'weather_encoded': 0,
         'tire_encoded': 1, 'pit_stops': 2},

        {'driver_name': 'Leclerc', 'team': 'Ferrari',
         'driver_skill': 92, 'car_performance': 92,
         'track_overtaking_difficulty': 95, 'track_avg_speed': 60,
         'grid_position': 3, 'weather_encoded': 0,
         'tire_encoded': 2, 'pit_stops': 2},

        {'driver_name': 'Norris', 'team': 'McLaren',
         'driver_skill': 90, 'car_performance': 88,
         'track_overtaking_difficulty': 95, 'track_avg_speed': 60,
         'grid_position': 4, 'weather_encoded': 0,
         'tire_encoded': 1, 'pit_stops': 2},

        {'driver_name': 'Alonso', 'team': 'AstonMartin',
         'driver_skill': 89, 'car_performance': 85,
         'track_overtaking_difficulty': 95, 'track_avg_speed': 60,
         'grid_position': 5, 'weather_encoded': 0,
         'tire_encoded': 1, 'pit_stops': 2},
    ]

    results = predictor.predict_race(monaco_race)
    print(results.to_string(index=False))

    # Example 2: Compare different tracks
    print("\n2. VERSTAPPEN WIN PROBABILITY BY TRACK (P1 Grid)")
    print("-" * 40)

    tracks = [
        ('Monaco (Street, Hard to Overtake)', 95, 60),
        ('Spa (High Speed, Easy Overtake)', 40, 85),
        ('Monza (Temple of Speed)', 35, 90),
        ('Singapore (Street, Night Race)', 70, 65),
        ('Silverstone (High Speed Corners)', 45, 82),
    ]

    base = {
        'driver_name': 'Verstappen', 'team': 'RedBull',
        'driver_skill': 95, 'car_performance': 95,
        'grid_position': 1, 'weather_encoded': 0,
        'tire_encoded': 2, 'pit_stops': 2
    }

    for track_name, overtaking, avg_speed in tracks:
        base['track_overtaking_difficulty'] = overtaking
        base['track_avg_speed'] = avg_speed
        prob = predictor.predict_probability(base)
        print(f"  {track_name}: {prob:.1%}")

    # Example 3: Weather impact
    print("\n3. WEATHER IMPACT ON WIN PROBABILITY")
    print("-" * 40)

    drivers = [
        ('Verstappen (RedBull)', 95, 95, 1),
        ('Hamilton (Mercedes)', 94, 90, 2),
        ('Leclerc (Ferrari)', 92, 92, 3),
    ]

    for weather in ['Dry', 'Mixed', 'Wet']:
        print(f"\n  {weather} Conditions:")
        weather_enc = {'Dry': 0, 'Mixed': 1, 'Wet': 2}[weather]

        for name, skill, car, grid in drivers:
            scenario = {
                'driver_name': name.split()[0], 'team': name.split()[1].strip('()'),
                'driver_skill': skill, 'car_performance': car,
                'track_overtaking_difficulty': 50, 'track_avg_speed': 80,
                'grid_position': grid, 'weather_encoded': weather_enc,
                'tire_encoded': 2, 'pit_stops': 2
            }
            prob = predictor.predict_probability(scenario)
            print(f"    {name}: {prob:.1%}")

    # Example 4: Grid position analysis
    print("\n4. GRID POSITION IMPACT (Verstappen)")
    print("-" * 40)

    for grid in range(1, 11):
        scenario = {
            'driver_name': 'Verstappen', 'team': 'RedBull',
            'driver_skill': 95, 'car_performance': 95,
            'track_overtaking_difficulty': 50, 'track_avg_speed': 80,
            'grid_position': grid, 'weather_encoded': 0,
            'tire_encoded': 2, 'pit_stops': 2
        }
        prob = predictor.predict_probability(scenario)
        bar = '#' * int(prob * 20)
        print(f"  P{grid:2d}: {prob:5.1%} {bar}")

    print("\n" + "=" * 60)
    print("Demo complete! Use f1_win_predictor.py to retrain the model.")
    print("=" * 60)


if __name__ == "__main__":
    main()
