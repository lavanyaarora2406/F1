# 🏎️ F1 Win Probability Predictor using Machine Learning

A Machine Learning project that predicts the **probability of a Formula 1 driver winning a race** using driver performance, car performance, track characteristics, weather conditions, tire strategy, and race strategy.

The project generates realistic synthetic Formula 1 race data, trains multiple classification models, and predicts win probabilities for different race scenarios.

---

# 📌 Project Overview

This project simulates Formula 1 races and uses Machine Learning to estimate each driver's chances of winning based on various racing factors.

The complete workflow includes:

* 🏁 Synthetic F1 data generation
* 📊 Exploratory data analysis
* 🧹 Data preprocessing
* 🤖 Training multiple Machine Learning models
* 📈 Model evaluation
* 💾 Model serialization
* 🔮 Win probability prediction

---

# 📂 Project Structure

```text
F1-Win-Probability-Predictor/
│
├── f1_data_generator.py
├── f1_win_predictor.py
├── f1_prediction_demo.py
├── f1_racing_data.csv
├── f1_win_predictor.pkl
├── README.md
```

---

# ✨ Features

* Synthetic Formula 1 race data generation
* Realistic race simulation
* Multiple Machine Learning classification models
* Driver win probability prediction
* Race scenario comparison
* Track analysis
* Weather impact analysis
* Grid position analysis
* Easy-to-use prediction interface
* Model saving and loading using Joblib

---

# 📊 Dataset Features

The generated dataset contains realistic Formula 1 race information including:

### Driver Information

* Driver Name
* Team
* Driver Skill Rating

### Car Information

* Car Performance Rating

### Track Information

* Overtaking Difficulty
* Average Track Speed

### Race Information

* Starting Grid Position
* Weather Conditions
* Tire Strategy
* Planned Pit Stops

### Target Variable

* Race Winner (Win / No Win)

---

# 🤖 Machine Learning Models

The project supports multiple classification algorithms:

* Logistic Regression
* Random Forest Classifier
* Gradient Boosting Classifier ✅ *(Default & Recommended)*

---

# 🏆 Model Capabilities

The trained model predicts:

* Driver win probability
* Race outcome likelihood
* Performance under different weather conditions
* Impact of starting position
* Influence of tire strategy
* Track-specific winning chances

---

# ⚙️ Installation

## Clone the Repository

```bash
git clone https://github.com/your-username/F1-Win-Probability-Predictor.git

cd F1-Win-Probability-Predictor
```

---

## Install Dependencies

```bash
pip install pandas numpy scikit-learn joblib
```

or

```bash
pip install -r requirements.txt
```

---

# 🚀 Usage

## Step 1: Generate Dataset & Train Model

```bash
python f1_win_predictor.py
```

This script will:

* Generate synthetic Formula 1 race data
* Save the dataset
* Train multiple classification models
* Evaluate model performance
* Save the best-performing model
* Display sample race predictions

---

## Step 2: Run the Prediction Demo

```bash
python f1_prediction_demo.py
```

The demo showcases:

* Driver win probabilities
* Track comparison
* Weather comparison
* Grid position analysis
* Race simulations

---

## Step 3: Generate Only the Dataset

```bash
python f1_data_generator.py
```

This creates:

```text
f1_racing_data.csv
```

---

# 🧠 Features Used for Prediction

| Feature                     | Description                        |
| --------------------------- | ---------------------------------- |
| Driver Skill                | Driver performance rating (0–100)  |
| Car Performance             | Vehicle performance rating (0–100) |
| Track Overtaking Difficulty | Difficulty of overtaking (0–100)   |
| Track Average Speed         | Average speed around the circuit   |
| Grid Position               | Starting position (1–20)           |
| Weather                     | Dry, Mixed, Wet                    |
| Tire Strategy               | Hard, Medium, Soft, Mixed          |
| Pit Stops                   | Planned number of pit stops        |

---

# 📈 Sample Predictions

The model can answer questions such as:

* What is Verstappen's chance of winning from Pole Position?
* How does rain affect Hamilton's winning probability?
* Which track gives Ferrari the highest winning chance?
* How much does starting P10 reduce the probability of winning?

Example Output:

```text
==================================================
F1 WIN PROBABILITY PREDICTOR
==================================================

Monaco GP

Driver           Team           Win Probability

Verstappen       Red Bull             28.3%
Hamilton         Mercedes            22.1%
Leclerc          Ferrari             19.7%
Norris           McLaren             12.4%
Alonso           Aston Martin        10.5%
```

---

# 🔬 Synthetic Data Generation

The project creates realistic Formula 1 racing data by simulating:

* 20-driver race weekends
* Driver skill variations
* Team performance differences
* Qualifying sessions
* Grid positions
* Weather changes
* Tire strategies
* Safety cars
* Race incidents
* Pit stop strategies
* Official Formula 1 points system

---

# 💻 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Jupyter Notebook

---

# 📦 Project Files

| File                    | Description                        |
| ----------------------- | ---------------------------------- |
| `f1_data_generator.py`  | Generates synthetic F1 racing data |
| `f1_win_predictor.py`   | Model training and prediction      |
| `f1_prediction_demo.py` | Demonstrates model predictions     |
| `f1_racing_data.csv`    | Generated training dataset         |
| `f1_win_predictor.pkl`  | Saved trained ML model             |
| `README.md`             | Project documentation              |

---

# 🚀 Future Improvements

* Add XGBoost and LightGBM models
* Hyperparameter optimization
* Streamlit web application
* Interactive race simulator
* SHAP feature importance visualization
* Driver comparison dashboard
* Real-world Formula 1 data integration
* Docker support
* CI/CD using GitHub Actions

---

# 📷 Screenshots

You can add screenshots of:

* Dataset Preview
* Model Training Results
* Feature Importance
* Prediction Output
* Race Comparison Results

Example folder structure:

```text
images/
├── dataset.png
├── prediction.png
├── training.png
└── comparison.png
```

---

# 📄 License

This project is developed for **educational and demonstration purposes**.

---

# 👩‍💻 Author

**Lavanya Arora**

⭐ If you found this project useful, consider giving it a **Star** on GitHub!
