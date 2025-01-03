# tests/test_data_preprocessor.py

import pytest
import pandas as pd
from datapreprocessor import DataPreprocessor

@pytest.fixture
def sample_data():
    data = {
        'release_ball_direction_x': [1.0, 2.0, 3.0, None, 5.0],
        'release_ball_direction_z': [2.0, 3.0, None, 5.0, 6.0],
        'result': [0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    return df

def test_filter_columns(sample_data):
    preprocessor = DataPreprocessor(
        model_type="Tree Based Classifier",
        y_variable=["result"],
        ordinal_categoricals=[],
        nominal_categoricals=[],
        numericals=["release_ball_direction_x", "release_ball_direction_z"],
        mode="train"
    )
    filtered_df = preprocessor.filter_columns(sample_data)
    assert 'result' in filtered_df.columns
    assert 'release_ball_direction_x' in filtered_df.columns
    assert 'release_ball_direction_z' in filtered_df.columns
    assert filtered_df.shape == (5, 3)

def test_handle_missing_values(sample_data):
    preprocessor = DataPreprocessor(
        model_type="Tree Based Classifier",
        y_variable=["result"],
        ordinal_categoricals=[],
        nominal_categoricals=[],
        numericals=["release_ball_direction_x", "release_ball_direction_z"],
        mode="train",
        options={
            'handle_missing_values': {
                'numerical_strategy': {
                    'strategy': 'median',
                    'imputer': 'SimpleImputer'
                },
                'categorical_strategy': {}
            }
        }
    )
    X_train, X_test, y_train, y_test = preprocessor.split_dataset(sample_data.drop('result', axis=1), sample_data['result'])
    X_train_imputed, X_test_imputed = preprocessor.handle_missing_values(X_train, X_test)
    assert X_train_imputed.isnull().sum().sum() == 0
    assert X_test_imputed.isnull().sum().sum() == 0

def test_preprocess_train(sample_data):
    preprocessor = DataPreprocessor(
        model_type="Tree Based Classifier",
        y_variable=["result"],
        ordinal_categoricals=[],
        nominal_categoricals=[],
        numericals=["release_ball_direction_x", "release_ball_direction_z"],
        mode="train",
        options={
            'handle_missing_values': {
                'numerical_strategy': {
                    'strategy': 'median',
                    'imputer': 'SimpleImputer'
                },
                'categorical_strategy': {}
            },
            'encode_categoricals': {
                'ordinal_encoding': 'OrdinalEncoder',
                'nominal_encoding': 'OneHotEncoder',
                'handle_unknown': 'ignore'
            },
            'apply_scaling': {
                'method': 'StandardScaler'
            },
            'implement_smote': {
                'variant': 'SMOTE',
                'params': {
                    'k_neighbors': 5,
                    'sampling_strategy': 'auto'
                }
            }
        }
    )
    X_train, X_test, y_train, y_test, recommendations, X_test_inverse = preprocessor.preprocess_train(sample_data.drop('result', axis=1), sample_data['result'])
    assert X_train.shape[0] > 0
    assert 'result' in y_train.name
    assert recommendations is not None
