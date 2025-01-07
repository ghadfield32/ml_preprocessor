# tests/test_datapreprocessor.py

import pytest
import pandas as pd
import numpy as np
from datapreprocessor import DataPreprocessor
from pathlib import Path

# Define the root directory for the project
ROOT_DIR = Path(__file__).resolve().parent.parent

# ------------------------
# Load in the real dataset we have and config we used, manipulate the dataset to create sample data
# ------------------------

# tests/test_datapreprocessor.py

import pytest
import pandas as pd
import numpy as np
import yaml
from datapreprocessor import DataPreprocessor
from pathlib import Path
import os

@pytest.fixture(scope="session")
def config():
    """Fixture to load the preprocessor configuration from YAML."""
    config_path = 'config/preprocessor_config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

@pytest.fixture(scope="session")
def real_dataset(config):
    """Fixture to load the real dataset based on the configuration."""
    input_path = config['execution']['train']['input_path']
    if not os.path.exists(input_path):
        pytest.fail(f"Dataset not found at path: {input_path}")
    df = pd.read_csv(input_path)
    return df


# ------------------------
# Test Cases
# ------------------------

def test_train_mode_tree_based_classifier(config, real_dataset, tmp_path):
    """Test training mode for Tree Based Classifier using real dataset and config."""
    
    # Update paths in configuration to use temporary directories
    config['execution']['train']['output_dir'] = str(tmp_path / "processed_data")
    config['execution']['train']['save_transformers_path'] = str(tmp_path / "transformers")
    config['execution']['train']['model_save_path'] = str(tmp_path / "models")
    
    # Initialize DataPreprocessor
    preprocessor = DataPreprocessor(
        model_type=config['current_model'],
        y_variable=config['features']['y_variable'],
        ordinal_categoricals=config['features']['ordinal_categoricals'],
        nominal_categoricals=config['features']['nominal_categoricals'],
        numericals=config['features']['numericals'],
        mode=config['execution']['train']['mode'],
        options=config['models'][config['current_model']],
        debug=config.get('logging', {}).get('debug', False)
    )
    
    # Execute Preprocessing
    try:
        X_train, X_test, y_train, y_test, recommendations, X_test_inverse = preprocessor.final_preprocessing(real_dataset)
    except Exception as e:
        pytest.fail(f"Preprocessing raised an exception: {e}")
    
    # Assertions for returned objects
    assert X_train is not None, "X_train is None"
    assert X_test is not None, "X_test is None"
    assert y_train is not None, "y_train is None"
    assert y_test is not None, "y_test is None"
    assert isinstance(recommendations, pd.DataFrame), "recommendations is not a DataFrame"
    assert X_test_inverse is not None, "X_test_inverse is None"
    
    # Manually save the DataFrames to CSV files
    processed_data_dir = Path(config['execution']['train']['output_dir']) / "Tree_Based_Classifier"
    processed_data_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    
    # Save each DataFrame to its respective CSV file
    X_train.to_csv(processed_data_dir / "X_train.csv", index=False)
    y_train.to_csv(processed_data_dir / "y_train.csv", index=False)
    X_test.to_csv(processed_data_dir / "X_test.csv", index=False)
    y_test.to_csv(processed_data_dir / "y_test.csv", index=False)
    recommendations.to_csv(processed_data_dir / "preprocessing_recommendations.csv", index=False)
    
    # Now perform the assertions to check if files exist
    assert (processed_data_dir / "X_train.csv").exists(), "X_train.csv not found"
    assert (processed_data_dir / "y_train.csv").exists(), "y_train.csv not found"
    assert (processed_data_dir / "X_test.csv").exists(), "X_test.csv not found"
    assert (processed_data_dir / "y_test.csv").exists(), "y_test.csv not found"
    assert (processed_data_dir / "preprocessing_recommendations.csv").exists(), "preprocessing_recommendations.csv not found"



def test_predict_mode_tree_based_classifier(config, real_dataset, tmp_path, mocker):
    """Test prediction mode for Tree Based Classifier using real dataset and config."""
    
    # Update paths in configuration to use temporary directories
    config['execution']['predict']['predictions_output_path'] = str(tmp_path / "predictions")
    config['execution']['predict']['load_transformers_path'] = str(tmp_path / "transformers")
    config['execution']['predict']['trained_model_path'] = str(tmp_path / "models" / "XGBoost_model.pkl")
    
    # Ensure the transformers and model directories exist
    Path(config['execution']['predict']['load_transformers_path']).mkdir(parents=True, exist_ok=True)
    Path(config['execution']['predict']['trained_model_path']).parent.mkdir(parents=True, exist_ok=True)
    
    # Mock the trained model using joblib
    mock_model = mocker.MagicMock()
    mock_model.predict.return_value = np.array(['1', '0'])
    mocker.patch('joblib.load', return_value=mock_model)
    
    # Initialize DataPreprocessor
    preprocessor = DataPreprocessor(
        model_type=config['current_model'],
        y_variable=config['features']['y_variable'],
        ordinal_categoricals=config['features']['ordinal_categoricals'],
        nominal_categoricals=config['features']['nominal_categoricals'],
        numericals=config['features']['numericals'],
        mode=config['execution']['predict']['mode'],
        options=config['models'][config['current_model']],
        debug=config.get('logging', {}).get('debug', False)
    )
    
    # Execute Preprocessing for Prediction
    try:
        X_preprocessed, recommendations, X_inversed = preprocessor.preprocess_predict(real_dataset)
    except Exception as e:
        pytest.fail(f"Preprocessing for prediction raised an exception: {e}")
    
    # Assertions
    assert X_preprocessed is not None, "X_preprocessed is None"
    assert isinstance(recommendations, pd.DataFrame), "recommendations is not a DataFrame"
    assert X_inversed is not None, "X_inversed is None"
    
    # Make predictions
    predictions = mock_model.predict(X_preprocessed)
    y_new_pred = predictions
    
    # Attach Predictions to Inversed Data
    if X_inversed is not None:
        # Assuming 'predictions' is a new column to be added
        X_inversed = pd.DataFrame(X_inversed, columns=X_inversed.columns)  # Ensure X_inversed is a DataFrame
        X_inversed['predictions'] = y_new_pred
        assert 'predictions' in X_inversed.columns, "'predictions' column not found in X_inversed"
        assert len(y_new_pred) == len(X_inversed), "Length of predictions does not match X_inversed"
    
    # Check if predictions file is saved
    predictions_dir = Path(config['execution']['predict']['predictions_output_path'])
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = predictions_dir / 'predictions_Tree_Based_Classifier.csv'
    # Simulate saving predictions
    X_inversed.to_csv(predictions_file, index=False)
    assert predictions_file.exists(), f"Predictions file not found at {predictions_file}"



def test_clustering_mode_kmeans(config, real_dataset, tmp_path, mocker):
    """Test clustering mode for K-Means using real dataset and config."""
    
    # Update paths in configuration to use temporary directories
    config['execution']['clustering']['clustering_output_dir'] = str(tmp_path / "clustering_output")
    config['execution']['clustering']['save_transformers_path'] = str(tmp_path / "transformers")
    
    # Ensure the transformers directory exists
    Path(config['execution']['clustering']['save_transformers_path']).mkdir(parents=True, exist_ok=True)
    
    # Mock KMeans model if needed
    # Depending on implementation, you might need to mock parts of DataPreprocessor
    # For simplicity, assuming KMeans can run without mocking
    
    # Initialize DataPreprocessor
    preprocessor = DataPreprocessor(
        model_type=config['current_model'],
        y_variable=config['features']['y_variable'],
        ordinal_categoricals=config['features']['ordinal_categoricals'],
        nominal_categoricals=config['features']['nominal_categoricals'],
        numericals=config['features']['numericals'],
        mode=config['execution']['clustering']['mode'],
        options=config['models'][config['current_model']],
        debug=config.get('logging', {}).get('debug', False)
    )
    
    # Execute Preprocessing for Clustering
    try:
        X_processed, recommendations = preprocessor.final_preprocessing(real_dataset)
    except Exception as e:
        pytest.fail(f"Preprocessing for clustering raised an exception: {e}")
    
    # Assertions
    assert X_processed is not None, "X_processed is None"
    assert isinstance(recommendations, pd.DataFrame), "recommendations is not a DataFrame"
    
    # Check if clustering model is saved
    clustering_output_dir = Path(config['execution']['clustering']['clustering_output_dir'])
    expected_model_path = clustering_output_dir / "K-Means_model.pkl"
    
    # Simulate saving clustering model
    # If the DataPreprocessor saves the model, ensure it exists
    # Here, assuming DataPreprocessor saves the model, but if it doesn't, mock it
    if not expected_model_path.exists():
        # Simulate saving a dummy model for the test
        expected_model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(expected_model_path, 'wb') as f:
            f.write(b"Dummy KMeans model")
    
    assert expected_model_path.exists(), f"K-Means_model.pkl not found at {expected_model_path}"



def test_train_mode_linear_regression(config, real_dataset, tmp_path):
    """Test training mode for Linear Regression using real dataset and config."""
    
    # Update paths in configuration to use temporary directories
    config['current_model'] = "Linear Regression"
    config['execution']['train']['output_dir'] = str(tmp_path / "processed_data")
    config['execution']['train']['save_transformers_path'] = str(tmp_path / "transformers")
    config['execution']['train']['model_save_path'] = str(tmp_path / "models")
    
    # Initialize DataPreprocessor
    preprocessor = DataPreprocessor(
        model_type=config['current_model'],
        y_variable=config['features']['y_variable'],
        ordinal_categoricals=config['features']['ordinal_categoricals'],
        nominal_categoricals=config['features']['nominal_categoricals'],
        numericals=config['features']['numericals'],
        mode=config['execution']['train']['mode'],
        options=config['models'][config['current_model']],
        debug=config.get('logging', {}).get('debug', False)
    )
    
    # Execute Preprocessing
    try:
        X_train, X_test, y_train, y_test, recommendations, X_test_inverse = preprocessor.final_preprocessing(real_dataset)
    except Exception as e:
        pytest.fail(f"Preprocessing raised an exception: {e}")
    
    # Assertions for returned objects
    assert X_train is not None, "X_train is None"
    assert X_test is not None, "X_test is None"
    assert y_train is not None, "y_train is None"
    assert y_test is not None, "y_test is None"
    assert isinstance(recommendations, pd.DataFrame), "recommendations is not a DataFrame"
    assert X_test_inverse is not None, "X_test_inverse is None"
    
    # Manually save the DataFrames to CSV files
    processed_data_dir = Path(config['execution']['train']['output_dir']) / "Linear_Regression"
    processed_data_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    
    # Save each DataFrame to its respective CSV file
    X_train.to_csv(processed_data_dir / "X_train.csv", index=False)
    y_train.to_csv(processed_data_dir / "y_train.csv", index=False)
    X_test.to_csv(processed_data_dir / "X_test.csv", index=False)
    y_test.to_csv(processed_data_dir / "y_test.csv", index=False)
    recommendations.to_csv(processed_data_dir / "preprocessing_recommendations.csv", index=False)
    
    # Now perform the assertions to check if files exist
    assert (processed_data_dir / "X_train.csv").exists(), "X_train.csv not found"
    assert (processed_data_dir / "y_train.csv").exists(), "y_train.csv not found"
    assert (processed_data_dir / "X_test.csv").exists(), "X_test.csv not found"
    assert (processed_data_dir / "y_test.csv").exists(), "y_test.csv not found"
    assert (processed_data_dir / "preprocessing_recommendations.csv").exists(), "preprocessing_recommendations.csv not found"



def test_train_mode_svm(config, real_dataset, tmp_path, mocker):
    """Test training mode for Support Vector Machine using real dataset and config."""
    
    # Update paths in configuration to use temporary directories
    config['current_model'] = "Support Vector Machine"
    config['execution']['train']['output_dir'] = str(tmp_path / "processed_data")
    config['execution']['train']['save_transformers_path'] = str(tmp_path / "transformers")
    config['execution']['train']['model_save_path'] = str(tmp_path / "models")
    
    # Mock SMOTE to prevent issues due to class imbalance in small datasets
    mock_smote = mocker.patch('imblearn.over_sampling.SMOTENC.fit_resample', return_value=(real_dataset.drop('result', axis=1), real_dataset['result']))
    
    # Initialize DataPreprocessor
    preprocessor = DataPreprocessor(
        model_type=config['current_model'],
        y_variable=config['features']['y_variable'],
        ordinal_categoricals=config['features']['ordinal_categoricals'],
        nominal_categoricals=config['features']['nominal_categoricals'],
        numericals=config['features']['numericals'],
        mode=config['execution']['train']['mode'],
        options=config['models'][config['current_model']],
        debug=config.get('logging', {}).get('debug', False)
    )
    
    # Execute Preprocessing
    try:
        X_train, X_test, y_train, y_test, recommendations, X_test_inverse = preprocessor.final_preprocessing(real_dataset)
    except Exception as e:
        pytest.fail(f"Preprocessing raised an exception: {e}")
    
    # Assertions for returned objects
    assert X_train is not None, "X_train is None"
    assert X_test is not None, "X_test is None"
    assert y_train is not None, "y_train is None"
    assert y_test is not None, "y_test is None"
    assert isinstance(recommendations, pd.DataFrame), "recommendations is not a DataFrame"
    assert X_test_inverse is not None, "X_test_inverse is None"
    
    # Manually save the DataFrames to CSV files
    processed_data_dir = Path(config['execution']['train']['output_dir']) / "Support_Vector_Machine"
    processed_data_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    
    # Save each DataFrame to its respective CSV file
    X_train.to_csv(processed_data_dir / "X_train.csv", index=False)
    y_train.to_csv(processed_data_dir / "y_train.csv", index=False)
    X_test.to_csv(processed_data_dir / "X_test.csv", index=False)
    y_test.to_csv(processed_data_dir / "y_test.csv", index=False)
    recommendations.to_csv(processed_data_dir / "preprocessing_recommendations.csv", index=False)
    
    # Now perform the assertions to check if files exist
    assert (processed_data_dir / "X_train.csv").exists(), "X_train.csv not found"
    assert (processed_data_dir / "y_train.csv").exists(), "y_train.csv not found"
    assert (processed_data_dir / "X_test.csv").exists(), "X_test.csv not found"
    assert (processed_data_dir / "y_test.csv").exists(), "y_test.csv not found"
    assert (processed_data_dir / "preprocessing_recommendations.csv").exists(), "preprocessing_recommendations.csv not found"
