# DataPreprocessor

**DataPreprocessor** is a comprehensive Python package designed to streamline the data preprocessing workflow for various machine learning models. It offers functionalities such as handling missing values, outlier detection, feature encoding, scaling, and addressing class imbalances using techniques like SMOTE.

## 🧰 Features

- **Flexible Preprocessing Pipeline:** Customize preprocessing steps based on your model requirements.
- **Automatic Outlier Detection:** Utilize methods like Z-Score Filtering, IQR Filtering, and Isolation Forest.
- **Categorical Encoding:** Support for Ordinal and One-Hot Encoding.
- **Scaling Options:** Apply StandardScaler, MinMaxScaler, or RobustScaler.
- **Class Imbalance Handling:** Implement SMOTE and its variants for classification tasks.
- **Configuration Driven:** Easily adjust preprocessing steps via YAML configuration files.
- **Inverse Transformation:** Reconstruct original feature values from transformed data for interpretability.

## 🚀 Installation

Ensure you have **Python 3.6** or higher installed.

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/ghadfield32-ml_preprocessor.git
   cd ghadfield32-ml_preprocessor

    Install the Package:

pip install .

Alternatively, for development purposes:

pip install -e .

Install Dependencies:

    pip install -r requirements.txt

📝 Usage
1. Basic Example: Training a Model

from datapreprocessor import DataPreprocessor
import pandas as pd

# Load your dataset
df = pd.read_csv('path_to_your_dataset.csv')

# Initialize the DataPreprocessor
preprocessor = DataPreprocessor(
    model_type="Tree Based Classifier",
    y_variable=["target"],
    ordinal_categoricals=["ordinal_feature1", "ordinal_feature2"],
    nominal_categoricals=["nominal_feature1", "nominal_feature2"],
    numericals=["num_feature1", "num_feature2"],
    mode="train",
    options=your_config_options,  # Replace with your configuration dictionary
    debug=True
)

# Execute Preprocessing
X_train, X_test, y_train, y_test, recommendations, X_test_inverse = preprocessor.final_preprocessing(df)

2. Running Preprocessing in Predict Mode

from datapreprocessor import DataPreprocessor
import pandas as pd

# Load new data for prediction
new_data = pd.read_csv('path_to_new_data.csv')

# Initialize the DataPreprocessor in predict mode
preprocessor = DataPreprocessor(
    model_type="Tree Based Classifier",
    y_variable=["target"],
    ordinal_categoricals=["ordinal_feature1", "ordinal_feature2"],
    nominal_categoricals=["nominal_feature1", "nominal_feature2"],
    numericals=["num_feature1", "num_feature2"],
    mode="predict",
    options=your_config_options,  # Replace with your configuration dictionary
    debug=True
)

# Execute Preprocessing
X_preprocessed, recommendations, X_inversed = preprocessor.preprocess_predict(new_data)

3. Training a Clustering Model

from datapreprocessor import DataPreprocessor
import pandas as pd

# Load your dataset
df = pd.read_csv('path_to_your_dataset.csv')

# Initialize the DataPreprocessor for clustering
preprocessor = DataPreprocessor(
    model_type="K-Means",
    y_variable=[],  # K-Means is unsupervised
    ordinal_categoricals=[],
    nominal_categoricals=[],
    numericals=["num_feature1", "num_feature2"],
    mode="clustering",
    options=your_clustering_config_options,  # Replace with your configuration dictionary
    debug=True
)

# Execute Preprocessing
X_processed, recommendations = preprocessor.final_preprocessing(df)

📄 Configuration

Customize preprocessing steps via the preprocessor_config.yaml file located in the config/ directory. Adjust parameters like imputation strategies, encoding methods, scaling techniques, and SMOTE configurations based on your project needs.
Sample Configuration:

current_model: "Tree Based Classifier"

model_types:
  - "Tree Based Classifier"
  - "Logistic Regression"
  - "K-Means"
  - "Linear Regression"
  - "Tree Based Regressor"
  - "Support Vector Machine"

features:
  ordinal_categoricals: []
  nominal_categoricals: []
  numericals:
    - release_ball_direction_x
    - release_ball_direction_z
    - release_ball_direction_y
    - elbow_release_angle
    - elbow_max_angle
    - wrist_release_angle
    - wrist_max_angle
    - knee_release_angle
    - knee_max_angle
    - release_ball_speed
    - calculated_release_angle
    - release_ball_velocity_x
    - release_ball_velocity_y
    - release_ball_velocity_z
  y_variable:
    - result

execution:
  shared:
    config_path: '../../dataset/test/preprocessor_config/preprocessor_config.yaml'
    features_metadata_path: '../../dataset/test/features_info/features_metadata.pkl'
    plot_output_dir: '../../dataset/test/plots'
    dataset_output_dir: '../../dataset/test/data'
    normalize_debug: false
    normalize_graphs_output: false

  train:
    mode: train
    input_path: '../../dataset/test/data/final_ml_dataset.csv'
    output_dir: '../../dataset/test/processed_data'
    training_output_dir: '../../dataset/test/training_output'
    save_transformers_path: '../../dataset/test/transformers/'
    model_save_path: '../../dataset/test/models'
    normalize_debug: false
    normalize_graphs_output: false

  predict:
    mode: predict
    load_transformers_path: '../../dataset/test/transformers/'
    prediction_input_path: '../../dataset/test/data/final_ml_dataset.csv'
    trained_model_path: '../../dataset/test/models/XGBoost_model.pkl'
    predictions_output_path: '../../dataset/test/data'
    normalize_debug: true
    normalize_graphs_output: false

  clustering:
    mode: clustering
    clustering_input_path: '../../dataset/test/data/final_ml_dataset.csv'
    clustering_output_dir: '../../dataset/test/clustering_output'
    normalize_debug: false
    normalize_graphs_output: false
    save_transformers_path: '../../dataset/test/transformers/'

models:
  # Model-specific configurations...

🧪 Testing

Ensure that your preprocessing pipeline works as expected by running the unit tests using pytest.

    Navigate to the Project Root:

cd ghadfield32-ml_preprocessor

Run the Tests:

    pytest

Test Coverage

The tests/test_datapreprocessor.py module includes comprehensive tests covering:

    Training Mode: Ensures that the preprocessing pipeline correctly handles data preparation for training various models.
    Prediction Mode: Validates that new data is preprocessed consistently for making predictions.
    Clustering Mode: Checks the preprocessing steps specific to clustering algorithms like K-Means.
    Handling Missing Values: Verifies that missing data is imputed as per the configuration.
    Categorical Encoding: Ensures that ordinal and nominal categorical features are encoded correctly.

🛠️ Contributing

Contributions are welcome! Please follow these steps:

    Fork the Repository:

    Click the Fork button at the top right corner of the repository page.

    Create a Feature Branch:

git checkout -b feature/YourFeatureName

Commit Your Changes:

git commit -m "Add your descriptive commit message"

Push to the Branch:

    git push origin feature/YourFeatureName

    Open a Pull Request:

    Navigate to the original repository and click on New pull request.

📄 License

This project is licensed under the MIT License. See the LICENSE file for details.
Continuous Integration (CI) with GitHub Actions

To ensure that your tests run automatically on every push and pull request, integrate Continuous Integration using GitHub Actions.
Implementation Steps:

    Create CI Workflow File:

    Create a new file at .github/workflows/ci.yml with the following content:

name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:

    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e .

    - name: Run Tests
      run: |
        pytest

Commit and Push:

git add .github/workflows/ci.yml
git commit -m "Add CI workflow with GitHub Actions"
git push origin main

Verify CI Runs:

Navigate to the Actions tab in your GitHub repository to see the CI workflow in action.