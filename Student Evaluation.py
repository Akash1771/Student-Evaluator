import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Create a synthetic dataset
np.random.seed(42)
data = {
    'Attendance': np.random.uniform(0.5, 1.0, 100),
    'Study_Hours': np.random.uniform(1, 10, 100),
    'Socio_Economic_Status': np.random.choice(['Low', 'Medium', 'High'], 100),
    'Final_Grade': np.random.uniform(50, 100, 100)
}
df = pd.DataFrame(data)
df = pd.get_dummies(df, columns=['Socio_Economic_Status'], drop_first=True)

# Split the dataset into features and target variable
X = df.drop('Final_Grade', axis=1)
y = df['Final_Grade']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print performance metrics
print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

# Visualization
plt.scatter(y_test, y_pred, color='blue')
plt.plot([50, 100], [50, 100], color='red', linestyle='--')
plt.title('Actual vs Predicted Final Grades')
plt.xlabel('Actual Final Grades')
plt.ylabel('Predicted Final Grades')
plt.xlim(50, 100)
plt.ylim(50, 100)
plt.grid()
plt.show()
