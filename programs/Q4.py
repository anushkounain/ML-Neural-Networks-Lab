#Program 4: Glass Types based on Chemical Properties using KNN 
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the Dataset
# You can download 'glass.csv' from the UCI repository or Kaggle.
# Ensure the dataset has 9 chemical feature columns and a target column named 'Type'.
#url = "https://githubusercontent.com"
#df = pd.read_csv(url)

# Load health dataset into panda's data frame
df = pd.read_csv(r"C:/Users/91636/Downloads/MLNNlab/glass.data",header=None,sep=',')

# Display basic info about the dataset
print("Dataset Shape:", df.shape)
print(df.head())

# 2. Split into Features (X) and Target Label (y)
# Assuming the last column contains the glass type
X = df.iloc[:, :-1]  # Chemical features (RI, Na, Mg, Al, Si, K, Ca, Ba, Fe)
y = df.iloc[:, -1]   # Glass Type class

# 3. Split data into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Standardize the features (Crucial for distance-based algorithms like KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Initialize and Train the KNN Classifier
# We set k = 5 as a standard starting point
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

# 6. Make Predictions and Evaluate the Model
y_pred = knn.predict(X_test_scaled)

print(f"\nAccuracy with K={k}: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
