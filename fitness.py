# -*- coding: utf-8 -*-
import pandas as pd
df = pd.read_csv("/content/health_fitness_dataset.csv")
df

fitness= df[df["activity_type"] == "Running"]
fitness

fitness.info()

fitness['speed'] = fitness['daily_steps'] / fitness['duration_minutes']
fitness['max_heart_rate'] = 220 - fitness['age']
fitness['vo2_max'] = 15.3 * (fitness['max_heart_rate'] / fitness['resting_heart_rate']) #maximum oxygen uptake

columns_to_drop = ['date','participant_id','health_condition','smoking_status','activity_type','blood_pressure_diastolic','blood_pressure_systolic']
fitness = fitness.drop(columns=columns_to_drop)

fitness.head()

fitness['vo2_max'].value_counts()

# Map intensity levels to numerical values
intensity_map = {'Low': 1, 'Medium': 2, 'High': 3}
fitness['intensity'] = fitness['intensity'].map(intensity_map)
gender_map = {'M': 0, 'F': 1}
fitness['gender'] = fitness['gender'].map(gender_map)

intensity_map = {1: 3.5, 2: 7.0, 3: 10.0}
#intensity_map = {1: 8.0, 2: 10.0, 3: 12.5}

# Add a new column to the DataFrame for MET values
fitness['MET'] = fitness['intensity'].map(intensity_map)

# Calculate estimated calories burned using the MET formula
fitness['estimated_calories'] = fitness['MET'] * fitness['weight_kg'] * (fitness['duration_minutes'] / 60)

import matplotlib.pyplot as plt
fitness.hist(figsize=(14, 10), bins=30)
plt.suptitle("Feature Distributions", fontsize=12)
plt.show()

# Compute correlation matrix
correlation_matrix = fitness.corr().abs()

# Plot heatmap for correlation visualization
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

# Compute correlation with heart_rate
correlation = fitness.corr()["avg_heart_rate"].abs()

# Display the correlation values in ascending order
print(correlation)

import pandas as pd
import numpy as np

# Step 1: Calculate correlation matrix
correlation_matrix = fitness.corr()

# Step 2: Filter columns based on correlation with the target variable
target_correlation = correlation_matrix['avg_heart_rate'].abs()  # Absolute correlation values
threshold = 0.004
columns_to_drop = target_correlation[target_correlation < threshold].index

# Step 3: Drop columns with correlation less than the threshold
fitness = fitness.drop(columns=columns_to_drop)

fitness=fitness.drop('max_heart_rate',axis=1)
fitness=fitness.drop('hydration_level',axis=1)
fitness=fitness.drop('MET',axis=1)

fitness

# Compute correlation matrix
correlation_matrix = fitness.corr().abs()

# Plot heatmap for correlation visualization
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

count = fitness[(fitness["avg_heart_rate"] >= 160) & (fitness["avg_heart_rate"] <= 200)].shape[0]
print(f"Number of occurrences in the range 160-200 bpm: {count}")

import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df["avg_heart_rate"])
plt.title("Heart Rate Distribution")
plt.show()

from scipy.stats import zscore

df["z_score"] = zscore(df["avg_heart_rate"])
outliers = df[(df["z_score"] > 3) | (df["z_score"] < -3)]
print(f"Number of outliers: {len(outliers)}")

import numpy as np
import pandas as pd
from collections import Counter

# Define features and target
X = fitness.drop(columns=['avg_heart_rate'])  # Features
y = fitness['avg_heart_rate']  # Target

# Separate minority class (160-200 bpm)
df_160_179 = fitness[(fitness['avg_heart_rate'] >= 160) & (fitness['avg_heart_rate'] < 180)].copy()
df_180_200 = fitness[(fitness['avg_heart_rate'] >= 180) & (fitness['avg_heart_rate'] <= 200)].copy()

# Increase upsampling factors
upsample_factor_160_179 = 5  # Increased from 3
upsample_factor_180_200 = 15 # Increased from 6

# Calculate new sample sizes
num_samples_160_179 = len(df_160_179) * upsample_factor_160_179
num_samples_180_200 = len(df_180_200) * upsample_factor_180_200

# Function to add Gaussian noise while keeping 'age' and 'intensity' unchanged
def augment_data(df, num_samples, noise_factor=0.05):
    augmented_data = df.sample(n=num_samples, replace=True).copy()

    # Apply Gaussian noise only to numerical columns **excluding 'avg_heart_rate'**
    for col in df.columns:
        if col not in ['avg_heart_rate', 'age', 'intensity']:  # Exclude categorical and target column
            augmented_data[col] += np.random.normal(0, noise_factor * df[col].std(), size=num_samples)

    # Ensure 'age' remains an integer
    augmented_data['age'] = augmented_data['age'].round().astype(int)

    # Ensure 'intensity' remains categorical
    augmented_data['intensity'] = augmented_data['intensity'].astype(int)

    return augmented_data

# Apply augmentation
df_160_179_upsampled = augment_data(df_160_179, num_samples_160_179, noise_factor=0.05)
df_180_200_upsampled = augment_data(df_180_200, num_samples_180_200, noise_factor=0.05)  # More aggressive

# Combine original data with upsampled data
fitnessresampled = pd.concat([fitness, df_160_179_upsampled, df_180_200_upsampled], ignore_index=True)

# Print new distribution of avg_heart_rate
print("Before upsampling:", Counter(y))
print("After upsampling:", Counter(fitnessresampled['avg_heart_rate']))
print("Total samples after upsampling:", len(fitnessresampled))
print("Age column type:", fitnessresampled['age'].dtype)  # Check age is integer
print("Unique values in intensity column:", fitnessresampled['intensity'].unique())  # Ensure categorical

fitnessresampled

import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=fitnessresampled["avg_heart_rate"])
plt.title("Heart Rate Distribution")
plt.show()

fitnessresampled.hist(figsize=(12, 8), bins=30)
plt.suptitle("Feature Distributions", fontsize=15)
plt.show()

count = fitnessresampled[(fitnessresampled["duration_minutes"] >= 1) & (fitnessresampled["duration_minutes"] <= 20)].shape[0]
print(f"Number of occurrences in the range 160-200 bpm: {count}")

count = fitnessresampled[(fitnessresampled["avg_heart_rate"] >= 160) & (fitnessresampled["avg_heart_rate"] <= 180)].shape[0]
print(f"Number of occurrences in the range 160-200 bpm: {count}")

count = fitnessresampled[(fitnessresampled["avg_heart_rate"] >= 180) & (fitnessresampled["avg_heart_rate"] <= 200)].shape[0]
print(f"Number of occurrences in the range 160-200 bpm: {count}")

fitnessresampled=fitnessresampled.drop('calories_burned',axis=1)

fitnessresampled["calories"] = fitnessresampled["estimated_calories"]

df=fitnessresampled.drop('estimated_calories',axis=1)

# Compute correlation matrix
correlation_matrix = df.corr().abs()

# Plot heatmap for correlation visualization
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

import numpy as np
# Define steps per minute based on intensity
def estimate_steps(row):
    if row['intensity'] == 1:
        steps_per_min = np.random.randint(100, 120)
    elif row['intensity'] == 2:
        steps_per_min = np.random.randint(120, 140)
    else:  # intensity == 3
        steps_per_min = np.random.randint(140, 160)
    return row['duration_minutes'] * steps_per_min

# Apply to dataset
df['steps'] = df.apply(estimate_steps, axis=1)
df = df.drop(columns=['daily_steps'])

# Save updated dataset
df.to_csv("fitness_with_steps_during_exercise.csv", index=False)

import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

# Number of rows
num_rows = 10000

# Helper functions
def calculate_bmi(weight_kg, height_m=1.75):
    return weight_kg / (height_m ** 2)

def estimate_calories(duration, intensity, weight_kg, heart_rate):
    base = duration * intensity * (weight_kg / 10)
    adjustment = 1 + (heart_rate - 120) / 120  # Higher HR increases calorie burn
    return round(base * adjustment, 2)

def estimate_heart_rate(intensity):
    if intensity == 1:
        return random.randint(100, 130)
    elif intensity == 2:
        return random.randint(130, 150)
    else:
        return random.randint(150, 180)

def estimate_steps(duration, intensity):
    # Steps per minute based on intensity
    if intensity == 1:
        steps_per_min = random.randint(100, 120)
    elif intensity == 2:
        steps_per_min = random.randint(120, 140)
    else:
        steps_per_min = random.randint(140, 160)
    return duration * steps_per_min

def estimate_vo2_max(age):
    base_vo2 = 45 - (age * 0.3)
    return round(base_vo2 + np.random.normal(0, 3), 2)

# Generate data
data = {
    'age': np.random.randint(18, 66, num_rows),
    'weight_kg': np.round(np.random.normal(75, 15, num_rows), 2),
    'duration_minutes': np.random.randint(1, 20, num_rows),  # Duration 1–19
}

# Derived features
data['bmi'] = [round(calculate_bmi(w, 1.75), 2) for w in data['weight_kg']]
data['intensity'] = [
    1 if d <= 5 else (2 if d <= 12 else 3)
    for d in data['duration_minutes']
]

# Add some randomness to intensity
for i in range(num_rows):
    if data['duration_minutes'][i] <= 12 and random.random() < 0.15:
        data['intensity'][i] = min(3, data['intensity'][i] + 1)

data['avg_heart_rate'] = [estimate_heart_rate(i) for i in data['intensity']]
data['steps'] = [estimate_steps(d, i) for d, i in zip(data['duration_minutes'], data['intensity'])]
data['vo2_max'] = [estimate_vo2_max(a) for a in data['age']]
data['calories'] = [
    estimate_calories(d, i, w, h) for d, i, w, h in zip(
        data['duration_minutes'], data['intensity'], data['weight_kg'], data['avg_heart_rate']
    )
]

# Save to CSV
df = pd.DataFrame(data)
df.to_csv('fitness_data_with_steps.csv', index=False)

print(f"✅ Created {len(df)} rows with 'steps' instead of 'daily_steps'.")

import pandas as pd

# Load both Excel files
df1 = pd.read_csv('/content/fitness_data_with_steps.csv')
df2 = pd.read_csv('/content/fitness_with_steps_during_exercise.csv')

# Concatenate the rows
merged_df = pd.concat([df1, df2] , ignore_index=True)

# Save the merged Excel file
merged_df.to_csv('merged_file.csv', index=False)

import seaborn as sns
sns.histplot(merged_df["avg_heart_rate"], kde=True)
plt.title("avg_heart_rate Distribution")
plt.show()

from sklearn.model_selection import train_test_split
target_column = 'avg_heart_rate'
X =merged_df.drop(columns=[target_column])
Y = merged_df[target_column]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mse)  # Root Mean Squared Error
r2 = r2_score(Y_test, Y_pred)

# Print results
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

from sklearn.metrics import r2_score

r2_train = r2_score(Y_train, model.predict(X_train))
r2_test = r2_score(Y_test, model.predict(X_test))

print(f"R² on Training Data: {r2_train:.2f}")
print(f"R² on Test Data: {r2_test:.2f}")

import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 300, 500],
    'gamma': [0.1, 0.2, 0.3]
}

xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, Y_train)

print("Best Parameters:", grid_search.best_params_)

best_xgb = XGBRegressor(
    gamma=0.3,
    learning_rate=0.1,
    max_depth=7,
    n_estimators=500,
    objective='reg:squarederror',
    random_state=42
)

best_xgb.fit(X_train, Y_train)

Y_pred = best_xgb.predict(X_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = mse ** 0.5
r2 = r2_score(Y_test, Y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")

import joblib
joblib.dump(best_xgb, "best_xgb.pkl")

# Save feature names to a file
with open("feature_names.txt", "w") as f:
    f.write("\n".join(X_train.columns))

import gradio as gr
import joblib
import numpy as np
# Load the trained model
model = joblib.load("best_xgb.pkl")

# Define prediction function
def predict(age, weight_kg,duration_minutes, intensity, daily_steps, bmi, vo2_max, calories):
    input_data = np.array([[age, weight_kg,duration_minutes, intensity, daily_steps,bmi, vo2_max, calories]])
    prediction = model.predict(input_data)
    return f"Predicted Avg Heart Rate: {prediction[0]:.2f}"

# Create Gradio interface
inputs = [
    gr.Number(label="Age"),
    gr.Number(label="Weight (kg)"),
    gr.Number(label="duration_minutes"),
    gr.Number(label="Intensity"),
    gr.Number(label="daily_steps"),
    gr.Number(label="BMI"),
    gr.Number(label="vo2_max"),
    gr.Number(label="calories"),

]

gr.Interface(fn=predict, inputs=inputs, outputs="text", title="Heart Rate Prediction").launch()

import pandas as pd

# Step 1: Calculate max heart rate (theoretical maximum heart rate)
merged_df["max_hr"] = 220 - merged_df["age"]

# Step 2: Add max_play_time feature
# We'll use a coefficient to scale VO2 max to realistic play time in minutes
play_time_coefficient = 1.5  # You can tweak this value based on model performance or domain knowledge

merged_df["max_play_time"] = (
    merged_df["vo2_max"] * play_time_coefficient /
    (merged_df["intensity"] * (merged_df["avg_heart_rate"] / merged_df["max_hr"]))
)
merged_df=merged_df.drop(columns=['max_hr'])

# Compute correlation matrix
correlation_matrix = merged_df.corr().abs()

# Plot heatmap for correlation visualization
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

merged_df=merged_df.drop(columns=['avg_heart_rate'])
merged_df=merged_df.drop(columns=['intensity','vo2_max'])

import seaborn as sns
sns.histplot(merged_df["max_play_time"], kde=True)
plt.title("avg_heart_rate Distribution")
plt.show()

from sklearn.model_selection import train_test_split
target_column = 'max_play_time'
X =merged_df.drop(columns=[target_column])
Y = merged_df[target_column]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mse)  # Root Mean Squared Error
r2 = r2_score(Y_test, Y_pred)

# Print results
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

from sklearn.metrics import r2_score

r2_train = r2_score(Y_train, model.predict(X_train))
r2_test = r2_score(Y_test, model.predict(X_test))

print(f"R² on Training Data: {r2_train:.2f}")
print(f"R² on Test Data: {r2_test:.2f}")

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.1, 0.2]
}

search = RandomizedSearchCV(GradientBoostingRegressor(), param_distributions=param_grid, n_iter=10, cv=3)
search.fit(X_train, Y_train)
print("Best Parameters:", search.best_params_)

from sklearn.metrics import mean_squared_error, r2_score
gb_model_MT  = search.best_estimator_
Y_pred = gb_model_MT.predict(X_test)

print("MSE:", mean_squared_error(Y_test, Y_pred))
print("R² Score:", r2_score(Y_test, Y_pred))

import joblib
joblib.dump(gb_model_MT, "gb_model_MT.pkl")

import joblib

# Load the XGBRegressor
model = joblib.load("gb_model_MT.pkl")

import matplotlib.pyplot as plt

plt.scatter(Y_test, Y_pred, alpha=0.6)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
plt.xlabel("Actual Max playing time")
plt.ylabel("Predicted Max playing time")
plt.title("Predicted vs. Actual Max playing time")
plt.grid(True)
plt.show()

residuals = Y_test - Y_pred

plt.scatter(Y_pred, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Max playing time")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True)
plt.show()

import gradio as gr
import joblib
import numpy as np
# Load the trained model
model = joblib.load("gb_model_MT.pkl")

# Define prediction function
def predict(age, weight_kg,duration_minutes, intensity, avg_heart_rate, bmi, vo2_max, calories, steps):
    input_data = np.array([[age, weight_kg,duration_minutes, intensity, avg_heart_rate, bmi, vo2_max, calories, steps]])
    prediction = model.predict(input_data)
    return f"Predicted Max playing time: {prediction[0]:.2f}"

# Create Gradio interface
inputs = [
    gr.Number(label="Age"),
    gr.Number(label="Weight (kg)"),
    gr.Number(label="duration_minutes"),
    gr.Number(label="Intensity"),
    gr.Number(label="avg_heart_rate"),
    gr.Number(label="BMI"),
    gr.Number(label="vo2_max"),
    gr.Number(label="calories"),
    gr.Number(label="steps"),
]

gr.Interface(fn=predict, inputs=inputs, outputs="text", title="Max playing time Prediction").launch()