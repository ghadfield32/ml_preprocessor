# tests/test_datapreprocessor.py

import pytest
import pandas as pd
import yaml
import os
from datapreprocessor import DataPreprocessor

@pytest.fixture(scope='session')
def config():
    """
    Load the preprocessor configuration from YAML file.
    """
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'preprocessor_config.yaml')
    if not os.path.exists(config_path):
        pytest.skip(f"Configuration file not found at {config_path}. Skipping tests that require configuration.")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

@pytest.fixture(scope='session')
def dataset():
    """
    Load the actual dataset from the dataset directory.
    """
    data_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'test', 'data', 'test_ml_dataset.csv')
    if not os.path.exists(data_path):
        pytest.skip(f"Dataset not found at {data_path}. Skipping tests that require dataset.")
    df = pd.read_csv(data_path)
    return df

@pytest.fixture
def model_config(config):
    """
    Extract configuration for 'Tree Based Classifier'.
    Modify this fixture or create additional ones for other models as needed.
    """
    model_type = "Tree Based Classifier"
    return config.get('models', {}).get(model_type, {})

@pytest.fixture
def preprocessor(model_config):
    """
    Initialize the DataPreprocessor with the specified model configuration.
    """
    # Extract necessary fields from configuration
    model_type = "Tree Based Classifier"  # Modify as needed
    y_variable = model_config.get('y_variable', ['result'])
    ordinal_categoricals = model_config.get('features', {}).get('ordinal_categoricals', [])
    nominal_categoricals = model_config.get('features', {}).get('nominal_categoricals', [])
    numericals = model_config.get('features', {}).get('numericals', [])
    
    # Execution parameters
    mode = 'train'
    options = model_config
    
    # Initialize DataPreprocessor
    dp = DataPreprocessor(
        model_type=model_type,
        y_variable=y_variable,
        ordinal_categoricals=ordinal_categoricals,
        nominal_categoricals=nominal_categoricals,
        numericals=numericals,
        mode=mode,
        options=options,
        debug=True  # Enable debug for test verbosity
    )
    return dp

def test_filter_columns(dataset, preprocessor):
    """
    Test that the filter_columns method correctly filters the DataFrame.
    """
    filtered_df = preprocessor.filter_columns(dataset)
    expected_columns = preprocessor.numericals + preprocessor.ordinal_categoricals + preprocessor.nominal_categoricals + preprocessor.y_variable
    for col in expected_columns:
        assert col in filtered_df.columns, f"Expected column '{col}' not found in filtered DataFrame."
    assert filtered_df.shape[0] == dataset.shape[0], "Number of rows should remain unchanged after filtering."
    assert filtered_df.shape[1] == len(expected_columns), "Number of columns after filtering does not match expected."

def test_handle_missing_values(dataset, preprocessor):
    """
    Test that missing values are handled correctly.
    """
    # Filter columns first
    filtered_df = preprocessor.filter_columns(dataset)
    X = filtered_df.drop(preprocessor.y_variable, axis=1)
    y = filtered_df[preprocessor.y_variable].iloc[:, 0]
    
    # Split dataset
    X_train, X_test, y_train, y_test = preprocessor.split_dataset(X, y)
    
    # Handle missing values
    X_train_imputed, X_test_imputed = preprocessor.handle_missing_values(X_train, X_test)
    
    # Assert no missing values in imputed datasets
    assert X_train_imputed.isnull().sum().sum() == 0, "Imputed training set contains missing values."
    if X_test_imputed is not None:
        assert X_test_imputed.isnull().sum().sum() == 0, "Imputed testing set contains missing values."

def test_preprocess_train(dataset, preprocessor):
    """
    Test the full preprocessing pipeline in train mode.
    """
    # Execute preprocessing
    X_train, X_test, y_train, y_test, recommendations, X_test_inverse = preprocessor.preprocess_train(
        dataset.drop(preprocessor.y_variable, axis=1),
        dataset[preprocessor.y_variable].iloc[:, 0]
    )
    
    # Assertions
    assert isinstance(X_train, pd.DataFrame), "X_train should be a DataFrame."
    assert isinstance(X_test, pd.DataFrame), "X_test should be a DataFrame."
    assert isinstance(y_train, pd.Series), "y_train should be a Series."
    assert isinstance(y_test, pd.Series), "y_test should be a Series."
    assert not X_train.empty, "X_train should not be empty."
    assert not X_test.empty, "X_test should not be empty."
    assert 'result' in y_train.name, "y_train should contain the 'result' variable."
    assert isinstance(recommendations, pd.DataFrame), "Recommendations should be a DataFrame."
    # Further assertions can be added based on expected transformations

def test_pipeline_integrity(dataset, preprocessor):
    """
    Test that the preprocessing pipeline maintains data integrity.
    """
    # Execute preprocessing
    X_train, X_test, y_train, y_test, recommendations, X_test_inverse = preprocessor.preprocess_train(
        dataset.drop(preprocessor.y_variable, axis=1),
        dataset[preprocessor.y_variable].iloc[:, 0]
    )
    
    # Check that the number of samples is consistent after SMOTE
    assert X_train.shape[0] >= dataset.shape[0] * (1 - preprocessor.options.get('implement_smote', {}).get('params', {}).get('sampling_strategy', 'auto')), "SMOTE did not increase the number of samples as expected."
    
    # Verify that the recommendations contain expected information
    assert not recommendations.empty, "Recommendations DataFrame should not be empty."
    assert 'Preprocessing Reason' in recommendations.columns, "Recommendations DataFrame should contain 'Preprocessing Reason' column."

def test_inverse_transformation(dataset, preprocessor):
    """
    Test the inverse transformation functionality.
    """
    # Execute preprocessing
    X_train, X_test, y_train, y_test, recommendations, X_test_inverse = preprocessor.preprocess_train(
        dataset.drop(preprocessor.y_variable, axis=1),
        dataset[preprocessor.y_variable].iloc[:, 0]
    )
    
    # Ensure inverse transformation was successful
    assert X_test_inverse is not None, "Inverse-transformed DataFrame should not be None."
    assert 'release_ball_direction_x' in X_test_inverse.columns, "Inverse-transformed DataFrame missing 'release_ball_direction_x'."
    assert 'release_ball_direction_z' in X_test_inverse.columns, "Inverse-transformed DataFrame missing 'release_ball_direction_z'."
    # Further assertions can check the closeness of original and inverse-transformed data

def test_handle_outliers(dataset, preprocessor):
    """
    Test the outlier handling functionality.
    """
    # Filter and handle missing values
    filtered_df = preprocessor.filter_columns(dataset)
    X = filtered_df.drop(preprocessor.y_variable, axis=1)
    y = filtered_df[preprocessor.y_variable].iloc[:, 0]
    X_train, X_test, y_train, y_test = preprocessor.split_dataset(X, y)
    X_train_imputed, X_test_imputed = preprocessor.handle_missing_values(X_train, X_test)
    
    # Handle outliers
    X_train_outliers_handled, y_train_outliers_handled = preprocessor.handle_outliers(X_train_imputed, y_train)
    
    # Assertions
    assert X_train_outliers_handled.isnull().sum().sum() == 0, "Outlier handling introduced missing values."
    assert X_train_outliers_handled.shape[0] <= X_train_imputed.shape[0], "Outlier handling did not remove samples as expected."

def test_encode_categoricals(dataset, preprocessor):
    """
    Test the categorical encoding functionality.
    """
    # Modify dataset to include nominal and ordinal categoricals if not present
    # For demonstration, let's add a nominal categorical feature
    dataset['category_nominal'] = ['A', 'B', 'A', 'C', 'B']
    
    # Update preprocessor's nominal_categoricals
    preprocessor.nominal_categoricals.append('category_nominal')
    
    # Execute preprocessing
    X_train, X_test, y_train, y_test, recommendations, X_test_inverse = preprocessor.preprocess_train(
        dataset.drop(preprocessor.y_variable, axis=1),
        dataset[preprocessor.y_variable].iloc[:, 0]
    )
    
    # Assertions
    assert 'category_nominal' not in X_train.columns, "Nominal categorical feature should be encoded and not present as original."
    # Depending on encoding strategy, check for encoded columns
    # For OneHotEncoder, check presence of encoded columns like 'category_nominal_A', etc.
    # For OrdinalEncoder, check if 'category_nominal' is transformed
    if preprocessor.options.get('encode_categoricals', {}).get('nominal_encoding') == 'OneHotEncoder':
        encoded_cols = [col for col in X_train.columns if col.startswith('category_nominal_')]
        assert len(encoded_cols) > 0, "OneHotEncoder did not create encoded columns for 'category_nominal'."
    elif preprocessor.options.get('encode_categoricals', {}).get('nominal_encoding') == 'OrdinalEncoder':
        assert 'category_nominal' in X_train.columns, "OrdinalEncoder did not transform 'category_nominal'."
        assert X_train['category_nominal'].dtype in [float, int], "OrdinalEncoder did not convert 'category_nominal' to numerical type."

def test_handle_smote(dataset, preprocessor):
    """
    Test that SMOTE is correctly applied to balance the dataset.
    """
    # Execute preprocessing
    X_train, X_test, y_train, y_test, recommendations, X_test_inverse = preprocessor.preprocess_train(
        dataset.drop(preprocessor.y_variable, axis=1),
        dataset[preprocessor.y_variable].iloc[:, 0]
    )
    
    # Calculate class distribution after SMOTE
    class_counts = y_train.value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    
    assert class_counts[majority_class] == class_counts[minority_class], "SMOTE did not balance the classes as expected."
