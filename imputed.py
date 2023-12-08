import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# Load your CSV data into a DataFrame
df = pd.read_csv("fraud.csv")

# Identify missing values
missing_values = df.isnull().sum()

# Separate features and target variable
features = df.drop(columns=["isFraud", "isFlaggedFraud"])
target = df["isFraud"]

# Identify columns with missing values
columns_with_missing_values = missing_values[missing_values > 0].index

# Impute missing values using Random Forest
for column in columns_with_missing_values:
    # Create a mask for missing values
    mask_missing = df[column].isnull()
    
    # Create a mask for non-missing values
    mask_not_missing = ~mask_missing
    
    # Create a RandomForestRegressor
    rf_imputer = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Fit the model on non-missing values
    rf_imputer.fit(features[mask_not_missing], df[column][mask_not_missing])
    
    # Predict missing values
    imputed_values = rf_imputer.predict(features[mask_missing])
    
    # Fill missing values with predicted values
    df.loc[mask_missing, column] = imputed_values

# Verify that there are no more missing values
if df.isnull().sum().sum() == 0:
    print("Imputation successful.")
else:
    print("Imputation failed.")

# Now you can use your imputed DataFrame for further analysis or modeling.
