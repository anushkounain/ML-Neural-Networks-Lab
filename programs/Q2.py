#Program 2: Create a Linear Regression Model on fuel efficiency 
#The program uses pandas and sklearn. Please install these packages
#pip install pandas
#pip install -U scikit-learn
#pip install matplotlib


import pandas as pd
import matplotlib.pyplot as plt
import numpy as py
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load health dataset into panda's data frame
df = pd.read_csv(r"C:/Users/91636/Downloads/MLNNlab/auto-mpg.data",header=None,sep='\\s+')

print('Dataset Header',df.head)
print('Our dataframe Shape: ',df.shape)

#X = 2 * df.random.rand(100, 1)
#y = 4 + 3 * X + df.random.randn(100, 1)

# data (as pandas dataframes) 
#X = df.data 
#y = df.data.targets 
  
X = df.iloc[:, 2:3]
Y = df.iloc[:, 5:6]

print('X',X)
print('Y',Y)

# 2. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# 3. Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Make predictions on the test set
y_pred = model.predict(X_test)

# 5. Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- Model Results ---")
print(f"Slope (Coefficient): {model.coef_[0][0]:.4f}")
print(f"Intercept: {model.intercept_[0]:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R2) Score: {r2:.4f}")

# 6. Visualize the training results and regression line
plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, color="blue", label="Training Data")
plt.scatter(X_test, y_test, color="green", label="Testing Data")
plt.plot(X, model.predict(X), color="red", linewidth=2, label="Regression Line")
plt.title("Linear Regression Implementation")
plt.xlabel("Independent Variable (X)")
plt.ylabel("Dependent Variable (y)")
plt.legend()
plt.grid(True)
plt.show()
