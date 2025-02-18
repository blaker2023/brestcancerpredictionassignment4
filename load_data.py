import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pickle

# Load dataset
file_path = "breast-cancer-wisconsin.data"
column_names = [
    "ID", "Clump_Thickness", "Uniformity_Cell_Size", "Uniformity_Cell_Shape",
    "Marginal_Adhesion", "Single_Epithelial_Cell_Size", "Bare_Nuclei",
    "Bland_Chromatin", "Normal_Nucleoli", "Mitoses", "Class"
]

df = pd.read_csv(file_path, names=column_names, na_values="?", dtype=str)
df.drop(columns=["ID"], inplace=True)
df = df.apply(pd.to_numeric)

# Handle missing values by replacing them with the median
imputer = SimpleImputer(strategy="median")
df.iloc[:, :-1] = imputer.fit_transform(df.iloc[:, :-1])

# Split dataset into features and target
X = df.drop(columns=["Class"])
y = df["Class"].replace({2: 0, 4: 1})  # Convert class labels (2=benign, 4=malignant)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train best model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
with open("best_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model training complete. Model saved as best_model.pkl.")
