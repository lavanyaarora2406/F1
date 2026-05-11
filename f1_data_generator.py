"""
F1 Data Generator - Creates synthetic F1 racing data for ML training
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_f1_data(num_races=1000, seed=42):
    """
    Generate synthetic F1 racing data with realistic features.

    Features include:
    - Driver skill rating
    - Car performance rating
    - Track characteristics
    - Weather conditions
    - Starting position
    - Tire strategy
    - Race outcome (win/loss)
    """
    np.random.seed(seed)
    random.seed(seed)

    # Driver pool with varying skill levels
    drivers = [
        {'name': 'Verstappen', 'skill': 95},
        {'name': 'Hamilton', 'skill': 94},
        {'name': 'Leclerc', 'skill': 92},
        {'name': 'Norris', 'skill': 90},
        {'name': 'Alonso', 'skill': 89},
        {'name': 'Sainz', 'skill': 88},
        {'name': 'Perez', 'skill': 87},
        {'name': 'Russell', 'skill': 87},
        {'name': 'Piastri', 'skill': 85},
        {'name': 'Gasly', 'skill': 83},
        {'name': 'Stroll', 'skill': 82},
        {'name': 'Hulkenberg', 'skill': 82},
        {'name': 'Tsunoda', 'skill': 81},
        {'name': 'Albon', 'skill': 83},
        {'name': 'Ricciardo', 'skill': 84},
        {'name': 'Bottas', 'skill': 85},
        {'name': 'Zhou', 'skill': 79},
        {'name': 'Magnussen', 'skill': 80},
        {'name': 'Sargeant', 'skill': 77},
        {'name': 'DeVries', 'skill': 78},
    ]

    # Tracks with different characteristics
    tracks = [
        {'name': 'Monaco', 'overtaking_difficulty': 95, 'avg_speed': 60},
        {'name': 'Spa', 'overtaking_difficulty': 40, 'avg_speed': 85},
        {'name': 'Monza', 'overtaking_difficulty': 35, 'avg_speed': 90},
        {'name': 'Singapore', 'overtaking_difficulty': 70, 'avg_speed': 65},
        {'name': 'Suzuka', 'overtaking_difficulty': 50, 'avg_speed': 80},
        {'name': 'Silverstone', 'overtaking_difficulty': 45, 'avg_speed': 82},
        {'name': 'Bahrain', 'overtaking_difficulty': 55, 'avg_speed': 75},
        {'name': 'Barcelona', 'overtaking_difficulty': 60, 'avg_speed': 78},
        {'name': 'RedBullRing', 'overtaking_difficulty': 50, 'avg_speed': 83},
        {'name': 'Interlagos', 'overtaking_difficulty': 45, 'avg_speed': 77},
        {'name': 'Jeddah', 'overtaking_difficulty': 65, 'avg_speed': 88},
        {'name': 'Miami', 'overtaking_difficulty': 55, 'avg_speed': 76},
        {'name': 'Baku', 'overtaking_difficulty': 60, 'avg_speed': 80},
        {'name': 'Mexico', 'overtaking_difficulty': 50, 'avg_speed': 72},
        {'name': 'COTA', 'overtaking_difficulty': 48, 'avg_speed': 79},
    ]

    # Teams with car performance ratings
    teams = {
        'RedBull': {'performance': 95},
        'Mercedes': {'performance': 90},
        'Ferrari': {'performance': 92},
        'McLaren': {'performance': 88},
        'AstonMartin': {'performance': 85},
        'Alpine': {'performance': 80},
        'Williams': {'performance': 75},
        'AlfaRomeo': {'performance': 76},
        'Haas': {'performance': 74},
        'AlphaTauri': {'performance': 77},
    }

    # Driver to team mapping
    driver_teams = {
        'Verstappen': 'RedBull', 'Perez': 'RedBull',
        'Hamilton': 'Mercedes', 'Russell': 'Mercedes',
        'Leclerc': 'Ferrari', 'Sainz': 'Ferrari',
        'Norris': 'McLaren', 'Piastri': 'McLaren',
        'Alonso': 'AstonMartin', 'Stroll': 'AstonMartin',
        'Gasly': 'Alpine', 'Ocon': 'Alpine',
        'Albon': 'Williams', 'Sargeant': 'Williams',
        'Bottas': 'AlfaRomeo', 'Zhou': 'AlfaRomeo',
        'Magnussen': 'Haas', 'Hulkenberg': 'Haas',
        'Tsunoda': 'AlphaTauri', 'Ricciardo': 'AlphaTauri',
    }

    data = []

    for race_idx in range(num_races):
        # Select random track
        track = random.choice(tracks)

        # Weather conditions
        weather = random.choice(['dry', 'dry', 'dry', 'wet', 'mixed'])
        weather_factor = {'dry': 1.0, 'wet': 0.8, 'mixed': 0.9}[weather]

        # Select 20 drivers for this race
        race_drivers = random.sample(drivers, min(20, len(drivers)))

        # Calculate performance scores for each driver
        driver_scores = []
        for driver in race_drivers:
            team_name = driver_teams.get(driver['name'], 'Alpine')
            team_perf = teams.get(team_name, {'performance': 80})['performance']

            # Combined score: driver skill + car performance
            base_score = (driver['skill'] * 0.6 + team_perf * 0.4)

            # Add some randomness for race-day performance
            race_form = np.random.normal(0, 5)

            # Weather affects drivers differently (some are better in wet)
            if weather == 'wet':
                wet_bonus = np.random.normal(0, 3)
                base_score += wet_bonus

            total_score = base_score * weather_factor + race_form
            driver_scores.append({
                'driver': driver['name'],
                'team': team_name,
                'score': total_score,
                'skill': driver['skill'],
                'car_perf': team_perf
            })

        # Sort by score to determine finishing order
        driver_scores.sort(key=lambda x: x['score'], reverse=True)

        # Assign starting positions with some variation
        for i, ds in enumerate(driver_scores):
            # Qualifying performance
            qual_performance = np.random.normal(0, 3)
            grid_position = max(1, min(20, i + 1 + int(qual_performance)))

        # Generate race entries
        for i, ds in enumerate(driver_scores):
            # Grid position based on qualifying
            base_grid = i + 1
            grid_jitter = int(np.random.normal(0, 2))
            grid_position = max(1, min(20, base_grid + grid_jitter))

            # Tire strategy
            tire_strategy = random.choice(['soft', 'medium', 'hard', 'mixed'])
            stops = random.choices([1, 2, 3], weights=[0.2, 0.6, 0.2])[0]

            # Race incidents
            incident = random.random() < 0.08  # 8% chance of incident
            safety_car = random.random() < 0.15  # 15% chance of SC

            # Calculate final position
            position_advantage = (20 - grid_position) * 0.3  # Starting position matters
            skill_advantage = ds['skill'] * 0.4
            car_advantage = ds['car_perf'] * 0.3

            # Tire strategy impact
            tire_impact = {'soft': 1, 'medium': 2, 'hard': 3, 'mixed': 0}[tire_strategy]

            final_score = position_advantage + skill_advantage + car_advantage + ds['score'] * 0.5
            if incident:
                final_score -= 30  # Big penalty for incidents

            ds['grid'] = grid_position
            ds['tire'] = tire_strategy
            ds['stops'] = stops
            ds['incident'] = incident
            ds['safety_car'] = safety_car
            ds['final_score'] = final_score

        # Sort by final score to get finishing order
        driver_scores.sort(key=lambda x: x['final_score'], reverse=True)

        # Assign positions
        for i, ds in enumerate(driver_scores):
            ds['position'] = i + 1
            ds['won'] = 1 if i == 0 else 0
            ds['podium'] = 1 if i < 3 else 0
            ds['points_finish'] = 1 if i < 10 else 0

            # Points awarded
            points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
            ds['points'] = points_map.get(i + 1, 0)

            data.append({
                'race_id': race_idx + 1,
                'track': track['name'],
                'track_overtaking_difficulty': track['overtaking_difficulty'],
                'track_avg_speed': track['avg_speed'],
                'driver_name': ds['driver'],
                'team': ds['team'],
                'driver_skill': ds['skill'],
                'car_performance': ds['car_perf'],
                'weather': weather,
                'grid_position': ds['grid'],
                'tire_strategy': ds['tire'],
                'pit_stops': ds['stops'],
                'had_incident': ds['incident'],
                'safety_car': ds['safety_car'],
                'finishing_position': ds['position'],
                'won': ds['won'],
                'podium': ds['podium'],
                'points_finish': ds['points_finish'],
                'points_earned': ds['points']
            })

    df = pd.DataFrame(data)
    return df


def save_data(df, filename='f1_racing_data.csv'):
    """Save generated data to CSV"""
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} records to {filename}")
    return df


if __name__ == "__main__":
    print("Generating F1 racing data...")
    df = generate_f1_data(num_races=500)
    save_data(df)

    print("\nData Summary:")
    print(f"Total records: {len(df)}")
    print(f"Unique drivers: {df['driver_name'].nunique()}")
    print(f"Unique tracks: {df['track'].nunique()}")
    print(f"Win rate distribution:\n{df.groupby('driver_name')['won'].mean().sort_values(ascending=False).head(10)}")
