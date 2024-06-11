import pandas as pd

# Read the original CSV file into a DataFrame
df = pd.read_csv("./original_dataset_all.csv")

# Columns to be removed
columns_to_remove = ['Source', 'CAS Number', 'Name type', 'Index']

# Drop the specified columns
df_cleaned = df.drop(columns=columns_to_remove)

# Create fingerprints here
# ADD CODE HERE

# Save the cleaned DataFrame to a new CSV file
df_cleaned.to_csv("../biodegrad.csv", index=False)
