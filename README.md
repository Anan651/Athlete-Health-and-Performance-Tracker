# 🤖 Athlete Health and Performance Tracker – AI Module

This repository contains the **AI component** of the *Athlete Health and Performance Tracker* graduation project. It focuses on building machine learning models to predict:

* 📈 **Average Heart Rate** during exercise
* ⏱️ **Maximum Play Time** based on VO₂ Max and physical exertion

Both models are trained, optimized, evaluated, and deployed using **Python**, **Scikit-learn**, **XGBoost**, and **Gradio**.

---

## 🧠 AI Models Overview

### 1. Heart Rate Prediction Model

* **Goal**: Predict the average heart rate based on workout and physical features
* **Model Used**: XGBoost Regressor
* **Input Features**:

  * Age, Weight (kg), Duration (min), Intensity, Steps, BMI, VO₂ Max, Calories
* **Performance**:

  * High R² score and low error metrics (MAE, RMSE)
* **Deployment**: Gradio web app for real-time prediction

### 2. Max Play Time Prediction Model

* **Goal**: Estimate the maximum safe play duration for an athlete
* **Model Used**: Gradient Boosting Regressor
* **Input Features**:

  * Age, Weight, Duration, Intensity, Avg Heart Rate, BMI, VO₂ Max, Calories, Steps
* **Performance**:

  * Trained and tuned using RandomizedSearchCV with strong R² and low RMSE
* **Deployment**: Separate Gradio interface for field testing

---

## 📊 Dataset

* The dataset used for training and experimentation is from Kaggle:
  🔗 [FitLife: Health and Fitness Tracking Dataset](https://www.kaggle.com/datasets/jijagallery/fitlife-health-and-fitness-tracking-dataset)

* Additional features were engineered such as:

  * VO₂ Max estimation
  * MET-based calorie estimation
  * Steps per minute estimation based on exercise intensity

---

## 📂 Key Files

| File                | Description                                                             |
| ------------------- | ----------------------------------------------------------------------- |
| `fitness.py`        | Full AI pipeline: data preprocessing, training, evaluation, Gradio apps |
| `best_xgb.pkl`      | Trained XGBoost model for heart rate prediction                         |
| `gb_model_MT.pkl`   | Trained XGBoost model for max play time prediction            |
| `feature_names.txt` | Saved feature list for inference                                        |
| `merged_file.csv`   | Cleaned and engineered dataset used for model training                  |

---

## 🚀 How to Run Locally

1. **Clone the Repo**:

   ```bash
   git clone https://github.com/yourusername/athlete-ai-module.git
   cd athlete-ai-module
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Gradio App**:

   ```bash
   python fitness.py
   ```

---

## 🧪 Sample Prediction Inputs

**Heart Rate Model:**

```python
predict(age=22, weight_kg=70, duration_minutes=15, intensity=2, daily_steps=1800, bmi=22.8, vo2_max=45.2, calories=280)
```

**Max Play Time Model:**

```python
predict(age=22, weight_kg=70, duration_minutes=15, intensity=2, avg_heart_rate=160, bmi=22.8, vo2_max=45.2, calories=280, steps=1800)
```

---

## 🧰 Tools & Libraries

* Python (Pandas, NumPy)
* Scikit-learn, XGBoost
* Gradio
* Matplotlib, Seaborn

---

## 👤 Author

**Anan Ahmed**
Bachelor of Communication and Electronics Engineering – Helwan University

