import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load data from a CSV file
df = pd.read_csv('../biodegrad.csv')  # Replace 'your_file.csv' with your CSV file path

# Remove non-numeric columns and NaN values
non_numeric_columns = ['Substance Name', 'Smiles', "Fingerprint List"]  # List of non-numeric columns to remove
df.drop(columns=non_numeric_columns, inplace=True)
print(f"Columns after removing {non_numeric_columns}:", df.columns.tolist())

# Prepare data (excluding 'Class' as it is the target variable)
X = df.drop(columns=['Class'])  # Features
y = df['Class']  # Target

# Ensure all columns are numeric
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)  # Convert to numeric and fill NaNs with 0

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Perform feature selection using RFECV with a Decision Tree model
model = DecisionTreeClassifier(random_state=42)
rfecv = RFECV(estimator=model, step=1, cv=5, scoring='accuracy')  # 5-fold cross-validation, adjust scoring if needed
rfecv.fit(X_train, y_train)

# Get the optimal number of features
optimal_features = rfecv.n_features_
print("Optimal number of features:", optimal_features)

# Get the selected features
selected_features = X.columns[rfecv.support_]
print(f"Selected Features (top {optimal_features}):", selected_features.tolist())

# Transform the training and testing data to keep only the selected features
X_train_rfecv = rfecv.transform(X_train)
X_test_rfecv = rfecv.transform(X_test)

# Fit the model using the selected features
model.fit(X_train_rfecv, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test_rfecv)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
