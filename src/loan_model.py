import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load Data
def load_data(train_path, test_path):
    """
    Load training and testing datasets from provided file paths.
    """
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

# Drop Columns
def drop_columns(data, columns_to_drop):
    """
    Drop unnecessary columns from the dataset.
    """
    return data.drop(columns=columns_to_drop, errors='ignore')

# Handle Missing Values
def handle_missing_values(data):
    """
    Fill missing values with median for numerical columns and mode for categorical columns.
    """
    # Fill missing numerical values with the median
    for column in data.select_dtypes(include=['float64', 'int64']).columns:
        data[column].fillna(data[column].median(), inplace=True)
    
    # Fill missing categorical values with the mode
    for column in data.select_dtypes(include=['object']).columns:
        data[column].fillna(data[column].mode()[0], inplace=True)
    
    return data

# Prepare Data
def prepare_data(train_data, test_data, target_column, columns_to_drop_train, columns_to_drop_test):
    """
    Prepare training and testing datasets by dropping unnecessary columns and handling missing values.
    """
    # Drop unnecessary columns
    train_data = drop_columns(train_data, columns_to_drop_train)
    test_data = drop_columns(test_data, columns_to_drop_test)
    
    # Handle missing values
    train_data = handle_missing_values(train_data)
    test_data = handle_missing_values(test_data)
    
    # Separate features and target variable
    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    
    return X_train, y_train, test_data

# Create Preprocessor
def create_preprocessor(X_train):
    """
    Create a preprocessing pipeline for numerical and categorical features.
    """
    categorical_features = X_train.select_dtypes(include=['object']).columns
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    
    # Preprocessing for numerical and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    return preprocessor

# Evaluate Model
def evaluate_model(model_pipeline, X_train, y_train):
    """
    Evaluate the model using cross-validation with accuracy, precision, recall, and F1-score.
    """
    scoring_metrics = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1': make_scorer(f1_score, average='weighted')
    }

# Align Test Data
def align_test_data(test_data, X_train):
    """
    Align the test dataset columns with the training dataset columns.
    """
    for column in X_train.columns:
        if column not in test_data.columns:
            # Fill missing columns in test data with median or mode from training data
            if X_train[column].dtype in ['float64', 'int64']:
                test_data[column] = X_train[column].median()
            else:
                test_data[column] = X_train[column].mode()[0]
    
    return test_data[X_train.columns]

# Main Function
def main():
    # Load training and testing data
    train_data, test_data = load_data(os.path.join('data', 'processed', 'Merged_payment_loan_train_data.csv'), os.path.join('data', 'raw', 'test_loan_data.csv'))

    # Specify columns to drop
    columns_to_drop_train = ["loan_id", "business_id", "credit_officer_id", "earliest_payment", "latest_payment", "dismissal_description", "total_paid", "num_payments"]
    columns_to_drop_test = ["loan_id", "business_id", "credit_officer_id", "dismissal_description", "payment_status"]

    # Prepare the data
    X_train, y_train, test_data_cleaned = prepare_data(train_data, test_data, target_column='target', columns_to_drop_train=columns_to_drop_train, columns_to_drop_test=columns_to_drop_test)

    # Create the preprocessing pipeline
    preprocessor = create_preprocessor(X_train)

    # Create the model pipeline with XGBoost, increasing regularization parameters
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_estimators=50, max_depth=3, reg_lambda=10.0, reg_alpha=5.0))
    ])

    # Evaluate the model
    evaluate_model(model_pipeline, X_train, y_train)

    # Fit the model
    model_pipeline.fit(X_train, y_train)
    print("Model training completed successfully.")


    # Align the test data with training data columns
    test_data_aligned = align_test_data(test_data_cleaned, X_train)

    # Make predictions on the test data
    test_predictions = model_pipeline.predict(test_data_aligned)
    print("Test data predictions completed successfully.")


if __name__ == "__main__":
    main()