# tests/test_datapreprocessor.py

import pytest
import pandas as pd
import numpy as np
import yaml
from datapreprocessor import DataPreprocessor
from pathlib import Path
import os
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
import joblib

def construct_model_path(model_save_base_dir: Path, model_sub_type: str) -> Path:
    """Constructs a standardized path for saving/loading models, matching example usage."""
    model_dir = model_save_base_dir / model_sub_type.replace(" ", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'trained_model.pkl'
    return model_path

def construct_predictions_path(predictions_output_dir: Path, model_sub_type: str) -> Path:
    """Constructs a standardized path for saving predictions, matching example usage."""
    predictions_dir = predictions_output_dir / model_sub_type.replace(" ", "_")
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = predictions_dir / f'predictions_{model_sub_type.replace(" ", "_")}.csv'
    return predictions_file

def get_dummy_model(model_type: str):
    """Returns a dummy model based on the model_type."""
    if "Classifier" in model_type:
        return DummyClassifier(strategy="most_frequent")
    elif "Regressor" in model_type:
        return DummyRegressor(strategy="mean")
    elif "Support Vector Machine" in model_type:
        return SVC(probability=True)
    elif "K-Means" in model_type:
        return KMeans(n_clusters=3, random_state=42)
    else:
        return DummyClassifier()

@pytest.fixture(scope="session")
def config():
    """
    Fixture to load the preprocessor configuration from YAML.
    Adjust the path to where your 'preprocessor_config.yaml' lives.
    This matches how the example usage loads config in 'train_predict.py'.
    """
    # Example path adjustment:
    config_path = Path(__file__).resolve().parent.parent / 'config/preprocessor_config.yaml'
    if not config_path.exists():
        pytest.fail(f"Configuration file not found at path: {config_path}")
    with open(config_path, 'r') as file:
        loaded_config = yaml.safe_load(file)
    return loaded_config

@pytest.fixture(scope="session")
def real_dataset(config):
    """
    Fixture to load the real dataset based on the updated config structure.
    Matches how 'train_predict.py' loads it via 'load_dataset' (minus the logging).
    """
    base_data_dir = Path(config["paths"]["base_data_dir"])
    raw_data_file = config["paths"]["raw_data_file"]
    input_path = base_data_dir / raw_data_file
    
    if not input_path.exists():
        pytest.fail(f"Dataset not found at path: {input_path}")
    
    df = pd.read_csv(input_path)
    return df

@pytest.mark.parametrize("model_type", [
    "Tree Based Classifier",
    "Logistic Regression",
    "K-Means",
    "Linear Regression",
    "Tree Based Regressor",
    "Support Vector Machine"
])
def test_datapreprocessor_modes(model_type, config, real_dataset, tmp_path, mocker):
    """
    Parameterized test for different model types (and their subtypes).
    This test now closely mimics the example usage from train_predict.py:
      - train mode
      - predict mode
      - (optionally) clustering mode for K-Means
    """

    # 1. Gather model subtypes from config
    model_sub_types = config.get('model_sub_types', {}).get(model_type, [])
    if not model_sub_types:
        pytest.skip(f"No subtypes found for model type '{model_type}'. Skipping tests.")

    # 2. Loop through each subtype
    for model_sub_type in model_sub_types:
        
        # Grab the model-specific config (like handle_missing_values, etc.)
        model_config = config.get('models', {}).get(model_type, {})
        if not model_config:
            pytest.fail(f"No configuration found for model type '{model_type}'.")

        # Paths from config (example usage style)
        model_save_base_dir = tmp_path / "models"
        predictions_output_dir = tmp_path / "predictions"
        transformers_dir = tmp_path / "transformers"

        # 3. Handle 'clustering' mode only for K-Means 
        #    (just like the example usage does)
        if model_type == "K-Means":
            # Initialize DataPreprocessor in 'clustering' mode
            preprocessor_clustering = DataPreprocessor(
                model_type=model_type,
                y_variable=None,  # K-Means is unsupervised
                ordinal_categoricals=config['features'].get('ordinal_categoricals', []),
                nominal_categoricals=config['features'].get('nominal_categoricals', []),
                numericals=config['features'].get('numericals', []),
                mode='clustering',
                options=model_config,
                debug=True,
                normalize_debug=True,
                normalize_graphs_output=True,
                graphs_output_dir=tmp_path / "plots",
                transformers_dir=transformers_dir
            )

            # Call final_preprocessing in clustering mode
            try:
                X_processed, recommendations = preprocessor_clustering.final_preprocessing(real_dataset)
            except Exception as e:
                pytest.fail(f"Clustering preprocessing failed for {model_sub_type}: {e}")

            # Assertions
            assert X_processed is not None, "X_processed is None in clustering mode."
            assert isinstance(recommendations, pd.DataFrame), "Recommendations not a DataFrame in clustering mode."

            # Save the preprocessed data for verification
            X_processed_df = pd.DataFrame(X_processed, 
                                          columns=preprocessor_clustering.pipeline.get_feature_names_out())
            clustering_output_dir = tmp_path / "clustering_output" / model_sub_type.replace(" ", "_")
            clustering_output_dir.mkdir(parents=True, exist_ok=True)
            X_processed_df.to_csv(clustering_output_dir / "X_processed.csv", index=False)
            recommendations.to_csv(clustering_output_dir / "preprocessing_recommendations.csv", index=False)

            # Assertions to Check if Files are Saved
            assert (clustering_output_dir / "X_processed.csv").exists(), f"X_processed.csv not found for {model_sub_type}"
            assert (clustering_output_dir / "preprocessing_recommendations.csv").exists(), f"preprocessing_recommendations.csv not found for {model_sub_type}"

            # Done with clustering. Move to the next subtype.
            continue

        # =============== TRAIN MODE ===============
        preprocessor_train = DataPreprocessor(
            model_type=model_type,
            y_variable=config['features'].get('y_variable', None),
            ordinal_categoricals=config['features'].get('ordinal_categoricals', []),
            nominal_categoricals=config['features'].get('nominal_categoricals', []),
            numericals=config['features'].get('numericals', []),
            mode='train',
            options=model_config,
            debug=True,
            normalize_debug=True,
            normalize_graphs_output=True,
            graphs_output_dir=tmp_path / "plots",
            transformers_dir=transformers_dir
        )

        # 4. final_preprocessing in train mode
        try:
            X_train, X_test, y_train, y_test, recommendations, X_test_inverse = preprocessor_train.final_preprocessing(real_dataset)
        except Exception as e:
            pytest.fail(f"Training final_preprocessing failed for {model_sub_type}: {e}")

        # Basic assertions
        assert X_train is not None, f"X_train is None for {model_sub_type}"
        assert y_train is not None, f"y_train is None for {model_sub_type}"
        assert isinstance(recommendations, pd.DataFrame), f"recommendations is not a DataFrame for {model_sub_type}"
        # For classification/regression, we expect X_test, y_test, X_test_inverse
        assert X_test is not None, f"X_test is None for {model_sub_type}"
        assert y_test is not None, f"y_test is None for {model_sub_type}"
        assert X_test_inverse is not None, f"X_test_inverse is None for {model_sub_type}"

        # 5. Check that transformers.pkl is created
        transformers_file = transformers_dir / 'transformers.pkl'
        assert transformers_file.exists(), f"transformers.pkl not found after train mode for {model_sub_type}"

        # =============== MODEL TRAINING ===============
        # Instead of using MagicMock, instantiate a real dummy model
        model = get_dummy_model(model_type)

        # Fit the model on the training data
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            pytest.fail(f"Model fitting failed for {model_sub_type}: {e}")

        # Save the trained model to disk
        model_path = construct_model_path(model_save_base_dir, model_sub_type)
        try:
            joblib.dump(model, model_path)
        except Exception as e:
            pytest.fail(f"Failed to save trained model for {model_sub_type}: {e}")

        # Verify model is saved
        assert model_path.exists(), f"Trained model file not found at {model_path} for {model_sub_type}"

        # =============== PREDICT MODE ===============
        preprocessor_predict = DataPreprocessor(
            model_type=model_type,
            y_variable=config['features'].get('y_variable', None),
            ordinal_categoricals=config['features'].get('ordinal_categoricals', []),
            nominal_categoricals=config['features'].get('nominal_categoricals', []),
            numericals=config['features'].get('numericals', []),
            mode='predict',
            options=model_config,
            debug=True,
            normalize_debug=True,
            normalize_graphs_output=True,
            graphs_output_dir=tmp_path / "plots",
            transformers_dir=transformers_dir
        )

        # 6. final_preprocessing in predict mode
        try:
            X_preprocessed, recommendations_pred, X_inversed = preprocessor_predict.final_preprocessing(real_dataset)
        except Exception as e:
            pytest.fail(f"Prediction final_preprocessing failed for {model_sub_type}: {e}")

        # Basic assertions for predict mode
        assert X_preprocessed is not None, f"X_preprocessed is None for {model_sub_type}"
        assert isinstance(recommendations_pred, pd.DataFrame), f"recommendations_pred is not DataFrame for {model_sub_type}"
        assert X_inversed is not None, f"X_inversed is None for {model_sub_type}"

        # 7. Load the trained model
        try:
            loaded_model = joblib.load(model_path)
        except Exception as e:
            pytest.fail(f"Failed to load trained model for {model_sub_type}: {e}")

        # 8. Make Predictions
        try:
            predictions = loaded_model.predict(X_preprocessed)
        except Exception as e:
            pytest.fail(f"Prediction failed for {model_sub_type}: {e}")

        # 9. Attach predictions to the inversed data
        if len(predictions) == len(X_inversed):
            X_inversed = X_inversed.copy()  # To avoid SettingWithCopyWarning
            X_inversed['predictions'] = predictions
            assert 'predictions' in X_inversed.columns, f"'predictions' column not found in X_inversed for {model_sub_type}"
            assert len(predictions) == len(X_inversed), f"Length mismatch between predictions and X_inversed for {model_sub_type}"
        else:
            pytest.fail("Predictions length does not match inversed data length.")

        # 10. Save Predictions
        predictions_file = construct_predictions_path(predictions_output_dir, model_sub_type)
        try:
            X_inversed.to_csv(predictions_file, index=False)
        except Exception as e:
            pytest.fail(f"Failed to save predictions for {model_sub_type}: {e}")

        # Assertions to Check if Files are Saved
        assert predictions_file.exists(), f"Predictions file not found at {predictions_file} for {model_sub_type}"

        # End of test for this (model_type, model_sub_type) combo
