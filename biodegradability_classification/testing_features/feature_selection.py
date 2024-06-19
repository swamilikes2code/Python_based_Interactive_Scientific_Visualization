import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import Button, Paragraph, PreText, Div
from bokeh.layouts import layout

# Load data from a CSV file
df = pd.read_csv('../biodegrad.csv')  # Replace with your CSV file path

# Remove non-numeric columns and NaN values
non_numeric_columns = ['Substance Name', 'Smiles', 'Fingerprint List']  # Non-numeric columns
df.drop(columns=non_numeric_columns, inplace=True)

# Create UI elements
button = Button(label="Run Feature Selection", button_type="success")
result_text = PreText(text="", width=500, height=200)
selected_features_text = Paragraph(text="")
status_text = Div(text="Click the button to start feature selection.", width=500, height=50)

# Define the callback function for the button
def run_feature_selection():
    global df
    # Indicate that the feature selection is running
    status_text.text = "Feature selection is running. Please wait..."
    button.disabled = True  # Disable the button to prevent multiple clicks

    try:
        
        # Prepare data (excluding 'Class' as it is the target variable)
        X = df.drop(columns=['Class'])  # Features
        y = df['Class']  # Target

        # Ensure all columns are numeric
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)  # Convert to numeric and fill NaNs with 0

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Perform feature selection using RFECV with a Decision Tree model
        model = DecisionTreeClassifier(random_state=42)
        rfecv = RFECV(estimator=model, step=1, cv=5, scoring='accuracy')  # 5-fold cross-validation
        rfecv.fit(X_train, y_train)

        # Get the optimal number of features
        optimal_features = rfecv.n_features_

        # Get the selected features
        selected_features = X.columns[rfecv.support_]
        selected_features_text.text = f"Selected Features (top {optimal_features}): {', '.join(selected_features)}"

        # Transform the training and testing data to keep only the selected features
        X_train_rfecv = rfecv.transform(X_train)
        X_test_rfecv = rfecv.transform(X_test)

        # Fit the model using the selected features
        model.fit(X_train_rfecv, y_train)

        # Make predictions and evaluate the model
        y_pred = model.predict(X_test_rfecv)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # Update the result text
        result_text.text = f"Accuracy: {accuracy}\n\nClassification Report:\n{report}"
        status_text.text = "Feature selection completed successfully."
    except Exception as e:
        # If an error occurs, display an error message
        status_text.text = f"An error occurred: {str(e)}"
    finally:
        # Re-enable the button after the process is complete
        button.disabled = False

# Add callback to button
button.on_click(run_feature_selection)

# Layout the application
layout = column(button, status_text, selected_features_text, result_text)

# Add layout to curdoc
curdoc().add_root(layout)
curdoc().title = "Feature Selection with Decision Tree"

# To run the Bokeh application, save this script and run it with:
# bokeh serve --show your_script.py
