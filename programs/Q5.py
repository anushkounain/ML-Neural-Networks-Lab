#5. Construct a Decision Tree classifier to predict whether a bank customer will subscribe to a term deposit and evaluate the model using accuracy and precision. Dataset: Bank Marketing Dataset https://archive.ics.uci.edu/ml/datasets/bank+marketing Tools Used: Python, Scikit-learn, Matplotlib.

#Program 5: Decision Tree for a Bank Customer Subscription 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# -------------------------------------------------------------------------
# STEP 1: Create a Synthetic Bank Customer Dataset (Mocking UCI Bank Data)
# -------------------------------------------------------------------------
# In real scenarios, use: data = pd.read_csv("bank_marketing.csv")

#This will read all columns
#data = pd.read_csv(r"d:\Notes\MLandNeuralNetworks\MLandNNLabPrograms\bank-full.csv",sep=';')


# Load only specific columns
data = pd.read_csv(r"C:/Users/91636/Downloads/MLNNlab/bank-full.csv",sep=';', usecols=['age', 'job','marital','balance','duration','campaign','y'])
print("--- Data Sample ---")
print(data.head)

np.random.seed(42)
num_customers = 1000


#data = pd.DataFrame({
#    'age': np.random.randint(18, 70, size=num_customers),
#    'job': np.random.choice(['blue-collar', 'management', 'technician', 'services'], size=num_customers),
#    'marital': np.random.choice(['married', 'single', 'divorced'], size=num_customers),
#    'balance': np.random.randint(-500, 5000, size=num_customers),
#    'duration': np.random.randint(10, 1000, size=num_customers), # Call duration in seconds
#    'campaign': np.random.randint(1, 6, size=num_customers),     # Number of contacts
#    'subscribed': np.random.choice(['no', 'yes'], size=num_customers, p=[0.85, 0.15]) # Target
#})
#print("--- Original Data Sample ---")
#print(data.head())

# -------------------------------------------------------------------------
# STEP 2: Preprocessing & Categorical Encoding
# -------------------------------------------------------------------------
# Decision trees require numeric data. We convert text categories using Label Encoding.
print(data.columns)

label_encoders = {}
categorical_cols = ['job', 'marital']

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Map target variable ('yes' -> 1, 'no' -> 0)
# We will map column 'y' to subscribed
data['y'] = data['y'].map({'yes': 1, 'no': 0})

# Separate Features (X) and Target (y)
X = data.drop(columns=['y'])
y = data['y']

# -------------------------------------------------------------------------
# STEP 3: Split into Training and Testing Sets
# -------------------------------------------------------------------------
# 80% data for training, 20% for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# -------------------------------------------------------------------------
# STEP 4: Build and Train the Decision Tree Classifier
# -------------------------------------------------------------------------
# Setting max_depth prevents the tree from growing infinitely and overfitting
# Gini impurity metric to find the best way to split data, alternative is entropy metric for faster execution
clf = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)

clf.fit(X_train, y_train)

# -------------------------------------------------------------------------
# STEP 5: Make Predictions and Evaluate Performance
# -------------------------------------------------------------------------
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\n--- Model Evaluation ---")
print(f"Accuracy Score: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))

# -------------------------------------------------------------------------
# STEP 6: Visualize the Decision Tree Structure
# -------------------------------------------------------------------------
plt.figure(figsize=(20, 10))
plot_tree(
    clf, 
    feature_names=X.columns, 
    class_names=['No', 'Yes'], 
    filled=True, 
    rounded=True, 
    fontsize=10
)
plt.title("Bank Customer Decision Tree Classifier Map", fontsize=16)
plt.show()

# -------------------------------------------------------------------------
# STEP 7: Check Feature Importance
# -------------------------------------------------------------------------
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': clf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n--- Feature Importance ---")
print(importance_df.to_string(index=False))
