"""
F1 Win Probability Predictor
A machine learning model that predicts the probability of an F1 driver winning a race.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')


class F1WinPredictor:
    """
    ML model to predict F1 driver win probability.
    """

    def __init__(self, model_type='gradient_boosting'):
        self.model_type = model_type
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = [
            'driver_skill',
            'car_performance',
            'track_overtaking_difficulty',
            'track_avg_speed',
            'grid_position',
            'weather_encoded',
            'tire_encoded',
            'pit_stops'
        ]

    def _encode_categorical(self, df, column, fit=True):
        """Encode categorical columns"""
        if fit:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))
            self.label_encoders[column] = le
        else:
            le = self.label_encoders.get(column)
            if le:
                known_classes = set(le.classes_)
                df[column] = df[column].apply(
                    lambda x: le.transform([x])[0] if x in known_classes else -1
                )
        return df

    def _preprocess(self, df, fit=False):
        """Preprocess the dataframe"""
        df = df.copy()

        # Encode weather if raw column exists
        if 'weather' in df.columns:
            weather_map = {'dry': 0, 'mixed': 1, 'wet': 2}
            df['weather_encoded'] = df['weather'].map(weather_map).fillna(0)
        elif 'weather_encoded' not in df.columns:
            df['weather_encoded'] = 0

        # Encode tire strategy if raw column exists
        if 'tire_strategy' in df.columns:
            tire_map = {'hard': 0, 'medium': 1, 'soft': 2, 'mixed': 3}
            df['tire_encoded'] = df['tire_strategy'].map(tire_map).fillna(1)
        elif 'tire_encoded' not in df.columns:
            df['tire_encoded'] = 1

        # Select features
        X = df[self.feature_columns].copy()

        # Handle any missing values
        X = X.fillna(X.median())

        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return pd.DataFrame(X_scaled, columns=self.feature_columns)

    def train(self, df):
        """
        Train the model on the provided data.

        Args:
            df: DataFrame with F1 racing data containing 'won' column
        """
        print("Preprocessing data...")
        X = self._preprocess(df, fit=True)
        y = df['won']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"Win rate in training: {y_train.mean():.2%}")

        # Select and train model
        if self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42
            )

        print(f"\nTraining {self.model_type} model...")
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        print("\n" + "=" * 50)
        print("MODEL EVALUATION")
        print("=" * 50)
        print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Loss', 'Win']))
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            print("\nFeature Importances:")
            importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(importance)

        return self

    def predict_probability(self, driver_data):
        """
        Predict win probability for a driver in a specific race scenario.

        Args:
            driver_data: dict with driver/race features

        Returns:
            float: Probability of winning (0-1)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        df = pd.DataFrame([driver_data])
        X = self._preprocess(df, fit=False)

        proba = self.model.predict_proba(X)[0, 1]
        return proba

    def predict_race(self, race_entries):
        """
        Predict win probabilities for all drivers in a race.

        Args:
            race_entries: list of dicts, one per driver

        Returns:
            DataFrame with drivers and their win probabilities
        """
        results = []
        for entry in race_entries:
            proba = self.predict_probability(entry)
            results.append({
                'driver': entry['driver_name'],
                'team': entry.get('team', 'Unknown'),
                'win_probability': proba
            })

        df = pd.DataFrame(results)
        df = df.sort_values('win_probability', ascending=False)
        return df

    def save_model(self, filepath='f1_win_predictor.pkl'):
        """Save the trained model"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath='f1_win_predictor.pkl'):
        """Load a trained model"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']
        self.feature_columns = data['feature_columns']
        print(f"Model loaded from {filepath}")
        return self


def create_sample_prediction(predictor):
    """Create sample predictions for demonstration"""
    print("\n" + "=" * 50)
    print("SAMPLE PREDICTIONS")
    print("=" * 50)

    # Sample race scenario
    sample_race = [
        {'driver_name': 'Verstappen', 'team': 'RedBull', 'driver_skill': 95, 'car_performance': 95,
         'track_overtaking_difficulty': 50, 'track_avg_speed': 80, 'grid_position': 1,
         'weather_encoded': 0, 'tire_encoded': 2, 'pit_stops': 2},

        {'driver_name': 'Hamilton', 'team': 'Mercedes', 'driver_skill': 94, 'car_performance': 90,
         'track_overtaking_difficulty': 50, 'track_avg_speed': 80, 'grid_position': 3,
         'weather_encoded': 0, 'tire_encoded': 1, 'pit_stops': 2},

        {'driver_name': 'Leclerc', 'team': 'Ferrari', 'driver_skill': 92, 'car_performance': 92,
         'track_overtaking_difficulty': 50, 'track_avg_speed': 80, 'grid_position': 2,
         'weather_encoded': 0, 'tire_encoded': 2, 'pit_stops': 2},

        {'driver_name': 'Norris', 'team': 'McLaren', 'driver_skill': 90, 'car_performance': 88,
         'track_overtaking_difficulty': 50, 'track_avg_speed': 80, 'grid_position': 4,
         'weather_encoded': 0, 'tire_encoded': 1, 'pit_stops': 2},

        {'driver_name': 'Alonso', 'team': 'AstonMartin', 'driver_skill': 89, 'car_performance': 85,
         'track_overtaking_difficulty': 50, 'track_avg_speed': 80, 'grid_position': 5,
         'weather_encoded': 0, 'tire_encoded': 1, 'pit_stops': 2},
    ]

    results = predictor.predict_race(sample_race)
    print("\nWin Probabilities for Sample Race:")
    print(results.to_string(index=False))

    # Individual scenario analysis
    print("\n" + "-" * 50)
    print("SCENARIO ANALYSIS: Verstappen")
    print("-" * 50)

    base_scenario = {
        'driver_name': 'Verstappen', 'team': 'RedBull',
        'driver_skill': 95, 'car_performance': 95,
        'track_overtaking_difficulty': 50, 'track_avg_speed': 80,
        'weather_encoded': 0, 'tire_encoded': 2, 'pit_stops': 2
    }

    scenarios = [
        ('Grid P1', {'grid_position': 1}),
        ('Grid P5', {'grid_position': 5}),
        ('Grid P10', {'grid_position': 10}),
        ('Wet Weather', {'weather_encoded': 2, 'grid_position': 1}),
        ('Wet from P5', {'weather_encoded': 2, 'grid_position': 5}),
    ]

    for name, changes in scenarios:
        scenario = {**base_scenario, **changes}
        prob = predictor.predict_probability(scenario)
        print(f"  {name}: {prob:.2%}")


def main():
    """Main function to run the full pipeline"""
    print("=" * 60)
    print("F1 WIN PROBABILITY PREDICTOR")
    print("=" * 60)

    # Generate data
    from f1_data_generator import generate_f1_data
    print("\nGenerating training data...")
    df = generate_f1_data(num_races=1000)
    print(f"Generated {len(df)} race entries")

    # Train model
    predictor = F1WinPredictor(model_type='gradient_boosting')
    predictor.train(df)

    # Sample predictions
    create_sample_prediction(predictor)

    # Save model
    predictor.save_model()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    return predictor


if __name__ == "__main__":
    predictor = main()
