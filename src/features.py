"""
Feature engineering for Telco Customer Churn.
Creates derived features from the cleaned dataset.
"""
import pandas as pd
import numpy as np


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps to the cleaned dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe (customerID dropped, Churn mapped to 0/1,
        TotalCharges converted to float, 11 blank rows dropped).
    
    Returns
    -------
    pd.DataFrame
        Dataframe with new engineered features and one-hot encoding applied.
    """
    df = df.copy()
    
    # --- Derived Features ---
    
    # tenure_group: Bucket tenure into lifecycle stages
    # Helps capture non-linear relationship between tenure and churn
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 72],
        labels=['0-12', '13-24', '25-48', '49-72']
    )
    
    # AvgCharge: Average monthly spend over customer lifetime
    # Reveals pricing pressure — high AvgCharge with short tenure = risk
    df['AvgCharge'] = df['TotalCharges'] / df['tenure']
    # Handle division by zero (tenure=0 should already be dropped, but safety net)
    df['AvgCharge'] = df['AvgCharge'].replace([np.inf, -np.inf], 0)
    df['AvgCharge'] = df['AvgCharge'].fillna(0)
    
    # TotalServices: Count of active add-on services
    # More services = higher switching cost = lower churn
    service_cols = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    df['TotalServices'] = df[service_cols].apply(
        lambda row: (row == 'Yes').sum(), axis=1
    )
    
    # IsMonthToMonth: Binary flag for month-to-month contracts
    # Month-to-month is the #1 churn driver by contract type
    df['IsMonthToMonth'] = (df['Contract'] == 'Month-to-month').astype(int)
    
    # IsFiber: Binary flag for Fiber optic internet
    # Fiber customers churn more — possibly price or service quality
    df['IsFiber'] = (df['InternetService'] == 'Fiber optic').astype(int)
    
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode all categorical columns with drop_first=True.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with engineered features.
    
    Returns
    -------
    pd.DataFrame
        Fully encoded numeric dataframe ready for modeling.
    """
    df = df.copy()
    
    # Identify categorical columns (object + category dtype)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # One-hot encode with drop_first to avoid multicollinearity
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Ensure all columns are numeric
    df = df.astype({col: int for col in df.select_dtypes(include='bool').columns})
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
    1. Convert TotalCharges from string to float (coerce blanks to NaN)
    2. Drop the 11 rows with blank TotalCharges (all have tenure=0)
    3. Drop customerID
    4. Map Churn Yes=1, No=0
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe loaded from CSV.
    
    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    df = df.copy()
    
    # Step 1: TotalCharges is object dtype with 11 blank-space values " "
    # pd.to_numeric with errors='coerce' turns them into NaN
    print(f"Shape before cleaning: {df.shape}")
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    n_null = df['TotalCharges'].isna().sum()
    print(f"TotalCharges NaN after coerce: {n_null}")
    
    # Step 2: Drop the 11 rows (all have tenure=0)
    df = df.dropna(subset=['TotalCharges'])
    print(f"Shape after dropping NaN TotalCharges: {df.shape}")
    
    # Step 3: Drop customerID (no predictive signal)
    df = df.drop(columns=['customerID'])
    print(f"Shape after dropping customerID: {df.shape}")
    
    # Step 4: Map Churn to binary
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    print(f"Churn distribution:\n{df['Churn'].value_counts()}")
    print(f"Final cleaned shape: {df.shape}")
    
    return df


def full_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the complete pipeline: clean -> feature engineer -> encode.
    """
    df = clean_data(df)
    df = create_features(df)
    df = encode_features(df)
    print(f"Final feature count: {df.shape[1] - 1} features + 1 target")
    return df
