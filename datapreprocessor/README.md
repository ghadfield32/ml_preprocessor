# ml_preprocessor
Automated preprocessing and inverse preprocessing utilities for machine learning models.

# **Comprehensive Guide to Enhanced Data Preprocessing Pipeline**

## Table of Contents

- [**Comprehensive Guide to Enhanced Data Preprocessing Pipeline**](#comprehensive-guide-to-enhanced-data-preprocessing-pipeline)
  - [Table of Contents](#table-of-contents)
  - [Overview of Goals and Steps](#overview-of-goals-and-steps)
    - [Primary Scenarios](#primary-scenarios)
    - [Preprocessing Steps](#preprocessing-steps)
  - [Model Types Classification](#model-types-classification)
  - [Detailed Step-by-Step Guide](#detailed-step-by-step-guide)
    - [1. Initialize Preprocessor and Configure Options](#1-initialize-preprocessor-and-configure-options)
    - [2. Split Dataset into Train/Test and X/y](#2-split-dataset-into-traintest-and-xy)
    - [3. Handle Missing Values](#3-handle-missing-values)
    - [4. Test for Normality](#4-test-for-normality)
    - [5. Handle Outliers](#5-handle-outliers)
    - [6. Choose and Apply Transformations (Based on Normality Tests)](#6-choose-and-apply-transformations-based-on-normality-tests)
    - [7. Encode Categorical Variables](#7-encode-categorical-variables)
    - [8. Apply Scaling (If Needed by Model)](#8-apply-scaling-if-needed-by-model)
    - [9. Implement SMOTE (Train Only)](#9-implement-smote-train-only)
    - [10. Train Model on Preprocessed Training Data](#10-train-model-on-preprocessed-training-data)
    - [11. Predict on Test Data (No SMOTE on Test)](#11-predict-on-test-data-no-smote-on-test)
    - [12. Final Inverse Transformations for Interpretability](#12-final-inverse-transformations-for-interpretability)
    - [13. Final Inverse Transformation Validation](#13-final-inverse-transformation-validation)
  - [Automated Output Datasets](#automated-output-datasets)
    - [Training Mode](#training-mode)
    - [Prediction Mode](#prediction-mode)
    - [Clustering Mode](#clustering-mode)
  - [SMOTE Calculations for Numericals](#smote-calculations-for-numericals)
  - [Final Verification and Readiness for K-Means Clustering](#final-verification-and-readiness-for-k-means-clustering)
- [**Usage Example**](#usage-example)
  - [**Contribution Guidelines**](#contribution-guidelines)
  - [**License**](#license)
  - [**Contact**](#contact)

---

## Overview of Goals and Steps

### Primary Scenarios

1. **Training and Testing Preparation:**
   - Preprocess data to obtain split datasets (`X_train_preprocessed`, `X_test_preprocessed`, `y_train`, `y_test`) ready for model training and evaluation.

2. **Prediction Preparation:**
   - Preprocess new or unseen data (`X_new_preprocessed`) using preprocessing assets derived from the training data to facilitate model predictions.

3. **Clustering Preparation:**
   - Preprocess data (`X_preprocessed`) tailored for unsupervised clustering algorithms like K-Means, ensuring appropriate scaling and encoding.

### Preprocessing Steps

1. **Initialize Preprocessor and Configure Options**
2. **Split Dataset into Train/Test and X/y**
3. **Handle Missing Values**
4. **Test for Normality**
5. **Handle Outliers**
6. **Choose and Apply Transformations (Based on Normality Tests)**
7. **Encode Categorical Variables**
8. **Apply Scaling (If Needed by Model)**
9. **Implement SMOTE (Train Only)**
10. **Train Model on Preprocessed Training Data**
11. **Predict on Test Data (No SMOTE on Test)**
12. **Final Inverse Transformations for Interpretability**
13. **Final Inverse Transformation Validation**

---

## Model Types Classification

Before delving into each preprocessing step, it's essential to categorize common model types to understand their preprocessing requirements better:

| **Model Type**                   | **Category**         | **Requires y** | **Notes**                                                                                   |
|----------------------------------|----------------------|-----------------|---------------------------------------------------------------------------------------------|
| Linear Regression                | Regression           | Yes             | Assumes linearity, normality, etc.                                                          |
| Logistic Regression              | Classification       | Yes             | Binary or multi-class classification                                                       |
| Decision Trees                   | Tree-Based           | Yes             | Scale-invariant; handles categorical variables inherently                                 |
| Random Forest                    | Tree-Based           | Yes             | Ensemble of decision trees                                                                 |
| Gradient Boosting                | Tree-Based           | Yes             | Boosted ensemble of trees                                                                  |
| Support Vector Machines (SVM)    | SVM-Based            | Yes             | Sensitive to feature scaling                                                               |
| k-Nearest Neighbors (k-NN)        | Distance-Based       | Yes             | Highly sensitive to feature scaling and distance metrics                                   |
| Neural Networks                  | Neural Networks      | Yes             | Requires scaled data; sensitive to feature distribution                                   |
| Clustering Algorithms (e.g., K-Means) | Unsupervised   | No              | Does not require y; focuses on feature distribution and scaling                            |

---

## Detailed Step-by-Step Guide

We'll adjust each preprocessing step to accommodate supervised models (classification and regression) and unsupervised clustering algorithms like K-Means. This ensures preprocessing aligns with the requirements of different model types.

### 1. Initialize Preprocessor and Configure Options

**Goal:**
Set up the preprocessing pipeline with necessary configurations, including model type, operational mode, and preprocessing options.

**Requires y:**
- **Training/Testing Preparation:** Yes (for supervised models)
- **Prediction Preparation:** No

**Implementation Example:**

```python
import logging
from typing import Dict, List, Optional

class DataPreprocessor:
    def __init__(
        self,
        model_type: str,
        column_assets: Dict[str, List[str]],
        mode: str,  # 'train', 'predict', 'clustering'
        options: Optional[Dict] = None,
        debug: bool = False,
        normalize_debug: bool = False,
        normalize_graphs_output: bool = False,
        graphs_output_dir: str = './plots',
        transformers_dir: str = './transformers'
    ):
        """
        Initialize the DataPreprocessor with model type, column assets, and user-defined options.

        Args:
            model_type (str): Type of the machine learning model (e.g., 'Logistic Regression').
            column_assets (Dict[str, List[str]]): Dictionary containing lists of columns for different categories.
            mode (str): Operational mode ('train', 'predict', 'clustering').
            options (Optional[Dict]): User-defined options for preprocessing steps.
            debug (bool): General debug flag to control overall verbosity.
            normalize_debug (bool): Flag to display normalization plots.
            normalize_graphs_output (bool): Flag to save normalization plots.
            graphs_output_dir (str): Directory to save plots.
            transformers_dir (str): Directory to save/load transformers.
        """
        self.model_type = model_type
        self.column_assets = column_assets
        self.mode = mode.lower()
        if self.mode not in ['train', 'predict', 'clustering']:
            raise ValueError("Mode must be one of 'train', 'predict', or 'clustering'.")
        self.options = options or {}
        self.debug = debug
        self.normalize_debug = normalize_debug
        self.normalize_graphs_output = normalize_graphs_output
        self.graphs_output_dir = graphs_output_dir
        self.transformers_dir = transformers_dir

        # Define model categories for accurate processing
        self.model_category = self.map_model_type_to_category()

        if self.model_category == 'unknown':
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.error(f"Model category for '{self.model_type}' is unknown. Check your configuration.")
            raise ValueError(f"Model category for '{self.model_type}' is unknown. Check your configuration.")

        # Initialize y_variable based on mode and model category
        if self.mode in ['train', 'predict'] and self.model_category in ['classification', 'regression']:
            self.y_variable = column_assets.get('y_variable', [])
            if not self.y_variable:
                if self.mode == 'train':
                    raise ValueError("Target variable 'y_variable' must be specified for supervised models in train mode.")
                # In predict mode, y_variable might not be present
        else:
            # For 'clustering' mode or unsupervised prediction
            self.y_variable = []

        # Fetch feature lists
        self.ordinal_categoricals = column_assets.get('ordinal_categoricals', [])
        self.nominal_categoricals = column_assets.get('nominal_categoricals', [])
        self.numericals = column_assets.get('numericals', [])

        # Initialize other variables
        self.scaler = None
        self.transformer = None
        self.ordinal_encoder = None
        self.nominal_encoder = None
        self.preprocessor = None
        self.smote = None
        self.feature_reasons = {col: '' for col in self.ordinal_categoricals + self.nominal_categoricals + self.numericals}
        self.preprocessing_steps = []
        self.normality_results = {}
        self.features_to_transform = []
        self.nominal_encoded_feature_names = []
        self.final_feature_order = []

        # Initialize placeholders for clustering-specific transformers
        self.cluster_transformers = {}
        self.cluster_model = None
        self.cluster_labels = None
        self.silhouette_score = None

        # Define default thresholds for SMOTE recommendations
        self.imbalance_threshold = self.options.get('smote_recommendation', {}).get('imbalance_threshold', 0.1)
        self.noise_threshold = self.options.get('smote_recommendation', {}).get('noise_threshold', 0.1)
        self.overlap_threshold = self.options.get('smote_recommendation', {}).get('overlap_threshold', 0.1)
        self.boundary_threshold = self.options.get('smote_recommendation', {}).get('boundary_threshold', 0.1)

        self.pipeline = None  # Initialize pipeline

        # Configure logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
    
    def map_model_type_to_category(self):
        """
        Map the model_type to its corresponding category.

        Returns:
            str: Category of the model.
        """
        classification_models = [
            'Logistic Regression',
            'Decision Trees',
            'Random Forest',
            'Gradient Boosting',
            'SVM',
            'k-NN',
            'Neural Networks'
        ]
        regression_models = [
            'Linear Regression'
        ]
        tree_based_models = [
            'Decision Trees',
            'Random Forest',
            'Gradient Boosting'
        ]
        svm_based_models = [
            'SVM'
        ]
        distance_based_models = [
            'k-NN'
        ]
        neural_network_models = [
            'Neural Networks'
        ]
        unsupervised_models = [
            'Clustering Algorithms'
        ]

        if self.model_type in regression_models:
            return 'regression'
        elif self.model_type in classification_models:
            return 'classification'
        elif self.model_type in tree_based_models:
            return 'tree_based_models'
        elif self.model_type in svm_based_models:
            return 'svm_based'
        elif self.model_type in distance_based_models:
            return 'distance_based'
        elif self.model_type in neural_network_models:
            return 'neural_networks'
        elif self.model_type in unsupervised_models:
            return 'clustering'
        else:
            return 'unknown'
```

**Options:**

- **Adjust Split Ratios:** Modify `split_ratio` based on dataset size and model requirements.
- **Cross-Validation Integration:** Incorporate cross-validation techniques if needed.
- **Custom Preprocessing Options:** Allow users to pass custom options for each preprocessing step.

---

### 2. Split Dataset into Train/Test and X/y

**Goal:**
Divide the dataset into training and testing subsets to ensure unbiased model evaluation and prevent data leakage.

**Requires y:**
- **Yes for supervised models (classification and regression)**
- **No for unsupervised models like clustering**

**Default Paths Based on Model Type:**

| **Model Type**          | **Split Strategy**                                                                 |
|-------------------------|------------------------------------------------------------------------------------|
| Classification          | Stratified split using `train_test_split` with `stratify=y`. Ensures balanced class distribution. |
| Regression              | Simple random split using `train_test_split`. No need for stratification.          |
| Clustering (Unsupervised)| Random or domain-specific split; y is not utilized.                             |

**Implementation Example:**

```python
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple, Optional

class DataPreprocessor:
    # ... [Previous code]

    def split_dataset(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None, 
        debug: bool = False
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series]]:
        """
        Split the dataset into training and testing sets while retaining original indices.

        Args:
            X (pd.DataFrame): Features.
            y (Optional[pd.Series]): Target variable.
            debug (bool): Flag to control debug outputs.

        Returns:
            Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series]]: X_train, X_test, y_train, y_test
        """
        step_name = "Split Dataset into Train and Test"
        self.logger.info(f"Step: {step_name}")

        # Debugging Statements
        if debug:
            self.logger.debug(f"Before Split - X shape: {X.shape}")
            if y is not None:
                self.logger.debug(f"Before Split - y shape: {y.shape}")
            else:
                self.logger.debug("Before Split - y is None")

        # Determine splitting based on mode
        if self.mode == 'train' and self.model_category in ['classification', 'regression']:
            if self.model_category == 'classification':
                stratify = y if self.options.get('split_dataset', {}).get('stratify_for_classification', False) else None
                test_size = self.options.get('split_dataset', {}).get('test_size', 0.2)
                random_state = self.options.get('split_dataset', {}).get('random_state', 42)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    stratify=stratify,
                    random_state=random_state
                )
                if debug:
                    self.logger.debug("Performed stratified split for classification.")
            elif self.model_category == 'regression':
                test_size = self.options.get('split_dataset', {}).get('test_size', 0.2)
                random_state = self.options.get('split_dataset', {}).get('random_state', 42)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=random_state
                )
                if debug:
                    self.logger.debug("Performed random split for regression.")
        else:
            # For 'predict' and 'clustering' modes or other categories
            X_train = X.copy()
            X_test = None
            y_train = y.copy() if y is not None else None
            y_test = None
            self.logger.info(f"No splitting performed for mode '{self.mode}' or model category '{self.model_category}'.")

        self.preprocessing_steps.append("Split Dataset into Train and Test")

        # Keep Indices Aligned Through Each Step
        if X_test is not None and y_test is not None:
            # Sort both X_test and y_test by index
            X_test = X_test.sort_index()
            y_test = y_test.sort_index()
            if debug:
                self.logger.debug("Sorted X_test and y_test by index for alignment.")

        # Debugging: Log post-split shapes and index alignment
        if debug:
            self.logger.debug(f"After Split - X_train shape: {X_train.shape}, X_test shape: {X_test.shape if X_test is not None else 'N/A'}")
            if self.model_category == 'classification' and y_train is not None and y_test is not None:
                self.logger.debug(f"Class distribution in y_train:\n{y_train.value_counts(normalize=True)}")
                self.logger.debug(f"Class distribution in y_test:\n{y_test.value_counts(normalize=True)}")
            elif self.model_category == 'regression' and y_train is not None and y_test is not None:
                self.logger.debug(f"y_train statistics:\n{y_train.describe()}")
                self.logger.debug(f"y_test statistics:\n{y_test.describe()}")

        # Check index alignment
        if y_train is not None and X_train.index.equals(y_train.index):
            if debug:
                self.logger.debug("X_train and y_train indices are aligned.")
        else:
            self.logger.warning("X_train and y_train indices are misaligned.")

        if X_test is not None and y_test is not None and X_test.index.equals(y_test.index):
            if debug:
                self.logger.debug("X_test and y_test indices are aligned.")
        elif X_test is not None and y_test is not None:
            self.logger.warning("X_test and y_test indices are misaligned.")

        return X_train, X_test, y_train, y_test
```

**Options:**

- **Adjust Split Ratios:** Tailor `test_size` based on dataset size and specific model needs.
- **Cross-Validation:** Utilize cross-validation strategies like `KFold`, `StratifiedKFold` for more robust evaluation.
- **Custom Splits:** Implement custom splitting strategies for specialized scenarios, such as multi-time-step forecasting.

---

### 3. Handle Missing Values

**Goal:**
Impute missing values appropriately to maintain data integrity.

**Requires y:**
- **No** for standard imputation
- **Yes** if using supervised imputation methods (e.g., IterativeImputer)

**Default Paths Based on Model Type:**

| **Model Type**           | **Numerical Imputation** | **Categorical Imputation**           |
|--------------------------|--------------------------|--------------------------------------|
| Linear/Logistic Regression | Mean Imputation          | Mode Imputation                      |
| Tree-Based Models         | Median Imputation        | Constant "Missing" label             |
| SVM, k-NN, Neural Networks | Median Imputation        | Mode Imputation                      |
| Clustering Algorithms     | Median Imputation        | Mode or Constant "Missing"           |

**Implementation Example:**

```python
from sklearn.impute import SimpleImputer, KNNImputer
import pandas as pd
from typing import Tuple, Optional

class DataPreprocessor:
    # ... [Previous code]

    def handle_missing_values(
        self, 
        X_train: pd.DataFrame, 
        X_test: Optional[pd.DataFrame] = None, 
        debug: bool = False
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Handle missing values for numerical and categorical features based on user options.

        Args:
            X_train (pd.DataFrame): Training features.
            X_test (Optional[pd.DataFrame]): Testing features.
            debug (bool): Flag to control debug outputs.

        Returns:
            Tuple[pd.DataFrame, Optional[pd.DataFrame]]: Imputed X_train and X_test.
        """
        step_name = "Handle Missing Values"
        self.logger.info(f"Step: {step_name}")

        new_columns = []

        # Fetch user-defined imputation options or set defaults
        impute_options = self.options.get('handle_missing_values', {})
        numerical_strategy = impute_options.get('numerical_strategy', {})
        categorical_strategy = impute_options.get('categorical_strategy', {})

        # Numerical Imputation
        numerical_imputer = None
        if self.numericals:
            if self.model_category in ['regression', 'classification', 'clustering']:
                default_num_strategy = 'mean'  # For clustering, mean imputation is acceptable
            else:
                default_num_strategy = 'median'
            num_strategy = numerical_strategy.get('strategy', default_num_strategy)
            num_imputer_type = numerical_strategy.get('imputer', 'SimpleImputer')  # Can be 'SimpleImputer', 'KNNImputer', etc.

            if debug:
                self.logger.debug(f"Numerical Imputation Strategy: {num_strategy.capitalize()}, Imputer Type: {num_imputer_type}")

            # Initialize numerical imputer based on user option
            if num_imputer_type == 'SimpleImputer':
                numerical_imputer = SimpleImputer(strategy=num_strategy)
            elif num_imputer_type == 'KNNImputer':
                knn_neighbors = numerical_strategy.get('knn_neighbors', 5)
                numerical_imputer = KNNImputer(n_neighbors=knn_neighbors)
            else:
                self.logger.error(f"Numerical imputer type '{num_imputer_type}' is not supported.")
                raise ValueError(f"Numerical imputer type '{num_imputer_type}' is not supported.")

            # Fit and transform ONLY on X_train
            X_train[self.numericals] = numerical_imputer.fit_transform(X_train[self.numericals])
            self.numerical_imputer = numerical_imputer  # Assign to self for saving
            self.feature_reasons.update({col: self.feature_reasons.get(col, '') + f'Numerical: {num_strategy.capitalize()} Imputation | ' for col in self.numericals})
            new_columns.extend(self.numericals)

            if X_test is not None:
                # Transform ONLY on X_test without fitting
                X_test[self.numericals] = numerical_imputer.transform(X_test[self.numericals])

        # Categorical Imputation
        categorical_imputer = None
        all_categoricals = self.ordinal_categoricals + self.nominal_categoricals
        if all_categoricals:
            default_cat_strategy = 'most_frequent'
            cat_strategy = categorical_strategy.get('strategy', default_cat_strategy)
            cat_imputer_type = categorical_strategy.get('imputer', 'SimpleImputer')

            if debug:
                self.logger.debug(f"Categorical Imputation Strategy: {cat_strategy.capitalize()}, Imputer Type: {cat_imputer_type}")

            # Initialize categorical imputer based on user option
            if cat_imputer_type == 'SimpleImputer':
                categorical_imputer = SimpleImputer(strategy=cat_strategy)
            elif cat_imputer_type == 'ConstantImputer':
                fill_value = categorical_strategy.get('fill_value', 'Missing')
                categorical_imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
            else:
                self.logger.error(f"Categorical imputer type '{cat_imputer_type}' is not supported.")
                raise ValueError(f"Categorical imputer type '{cat_imputer_type}' is not supported.")

            # Fit and transform ONLY on X_train
            X_train[all_categoricals] = categorical_imputer.fit_transform(X_train[all_categoricals])
            self.categorical_imputer = categorical_imputer  # Assign to self for saving
            self.feature_reasons.update({
                col: self.feature_reasons.get(col, '') + (f'Categorical: Constant Imputation (Value={categorical_strategy.get("fill_value", "Missing")}) | ' if cat_imputer_type == 'ConstantImputer' else f'Categorical: {cat_strategy.capitalize()} Imputation | ')
                for col in all_categoricals
            })
            new_columns.extend(all_categoricals)

            if X_test is not None:
                # Transform ONLY on X_test without fitting
                X_test[all_categoricals] = categorical_imputer.transform(X_test[all_categoricals])

        self.preprocessing_steps.append("Handle Missing Values")

        # Completion Logging
        if debug:
            self.logger.debug(f"Completed: Handle Missing Values. Dataset shape after imputation: {X_train.shape}")
            self.logger.debug(f"Missing values after imputation in X_train:\n{X_train.isnull().sum()}")
            self.logger.debug(f"New columns handled: {new_columns}")
        else:
            self.logger.info(f"Step '{step_name}' completed.")

        return X_train, X_test
```

**Options:**

- **Advanced Imputers:**
  - Utilize `KNNImputer` or `IterativeImputer` for more sophisticated imputation scenarios.
  
    ```python
    from sklearn.impute import KNNImputer, IterativeImputer
    
    # Example: KNNImputer
    num_imputer = KNNImputer(n_neighbors=5)
    
    # Example: IterativeImputer
    num_imputer = IterativeImputer(random_state=self.random_state)
    ```
  
- **Model-Based Imputation:**
  - Leverage predictive models to estimate missing values based on other features.
  
- **Custom Strategies:**
  - Implement domain-specific logic for imputation as required by the dataset.
  
- **Persist Imputers:**
  - Save fitted imputers to ensure consistent application during prediction.
  
    ```python
    import pickle

    with open('num_imputer.pkl', 'wb') as f:
        pickle.dump(num_imputer, f)

    with open('cat_imputer.pkl', 'wb') as f:
        pickle.dump(cat_imputer, f)
    ```

---

### 4. Test for Normality

**Goal:**
Determine if feature distributions meet model assumptions regarding normality.

**Requires y:**
- **No**

**Default Paths Based on Model Type:**

| **Model Type**           | **Normality Testing**                                                                            |
|--------------------------|--------------------------------------------------------------------------------------------------|
| Linear/Logistic Regression | Use both p-value (e.g., Shapiro-Wilk test) and skewness. If p-value < 0.05 **OR** skewness > threshold, consider transformation. |
| Neural Networks, SVM, k-NN, Clustering | Use skewness only. If skewness > threshold, consider transformation.                           |
| Tree-Based Models         | No p-value usage. Only transform if skewness > threshold.                                      |

**Implementation Example:**

```python
from scipy.stats import shapiro, skew, anderson
import pandas as pd
from typing import Dict

class DataPreprocessor:
    # ... [Previous code]

    def test_normality(
        self, 
        X_train: pd.DataFrame, 
        debug: bool = False
    ) -> Dict[str, Dict]:
        """
        Test normality for numerical features based on normality tests and user options.

        Args:
            X_train (pd.DataFrame): Training features.
            debug (bool): Flag to control debug outputs.

        Returns:
            Dict[str, Dict]: Dictionary with normality test results for each numerical feature.
        """
        step_name = "Test for Normality"
        self.logger.info(f"Step: {step_name}")
        normality_results = {}

        # Fetch user-defined normality test options or set defaults
        normality_options = self.options.get('test_normality', {})
        p_value_threshold = normality_options.get('p_value_threshold', 0.05)
        skewness_threshold = normality_options.get('skewness_threshold', 1.0)
        additional_tests = normality_options.get('additional_tests', [])  # e.g., ['anderson-darling']

        for col in self.numericals:
            data = X_train[col].dropna()
            skewness = data.skew()
            kurtosis = data.kurtosis()

            # Determine which normality test to use based on sample size and user options
            test_used = 'Shapiro-Wilk'
            p_value = 0.0

            if len(data) <= 5000:
                stat, p_val = shapiro(data)
                test_used = 'Shapiro-Wilk'
                p_value = p_val
            else:
                result = anderson(data)
                test_used = 'Anderson-Darling'
                # Determine p-value based on critical values
                p_value = 0.0  # Default to 0
                for cv, sig in zip(result.critical_values, result.significance_level):
                    if result.statistic < cv:
                        p_value = sig / 100
                        break

            # Apply user-defined or default criteria
            if self.model_category in ['regression', 'classification', 'clustering']:
                # Linear, Logistic Regression, and Clustering: Use p-value and skewness
                needs_transform = (p_value < p_value_threshold) or (abs(skewness) > skewness_threshold)
            else:
                # Other models: Use skewness, and optionally p-values based on options
                use_p_value = normality_options.get('use_p_value_other_models', False)
                if use_p_value:
                    needs_transform = (p_value < p_value_threshold) or (abs(skewness) > skewness_threshold)
                else:
                    needs_transform = abs(skewness) > skewness_threshold

            normality_results[col] = {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'p_value': p_value,
                'test_used': test_used,
                'needs_transform': needs_transform
            }

            # Conditional Detailed Logging
            if debug:
                self.logger.debug(f"Feature '{col}': p-value={p_value:.4f}, skewness={skewness:.4f}, needs_transform={needs_transform}")

        self.normality_results = normality_results
        self.preprocessing_steps.append(step_name)

        # Completion Logging
        if debug:
            self.logger.debug(f"Completed: {step_name}. Normality results computed.")
        else:
            self.logger.info(f"Step '{step_name}' completed: Normality results computed.")

        return normality_results
```

**Options:**

- **Additional Normality Tests:**
  - Incorporate Anderson-Darling or Kolmogorov-Smirnov tests for more robust normality assessment.
  
    ```python
    from scipy.stats import anderson, kstest

    # Anderson-Darling Test
    result = anderson(data)
    # Handle based on critical values

    # Kolmogorov-Smirnov Test
    stat, p_val = kstest(data, 'norm')
    ```
  
- **Adjust Thresholds:**
  - Modify `p_value_threshold` or `skewness_threshold` based on specific model sensitivities or domain requirements.
  
- **Combine with Visualization:**
  - Use QQ plots or histograms for visual assessment alongside statistical tests.
  
    ```python
    import matplotlib.pyplot as plt
    from scipy.stats import probplot

    for col in features_to_transform:
        probplot(X[col], dist="norm", plot=plt)
        plt.title(f'QQ Plot of {col}')
        plt.show()
    ```
  
- **Considerations for Prediction Preparation:**
  - **Assumption of Consistency:** Assume that any transformations based on normality were already handled during training preprocessing. Therefore, skip testing normality on new prediction data.

---

### 5. Handle Outliers

**Goal:**
Reduce the influence of extreme values that can skew model performance.

**Requires y:**
- **No** for standard outlier handling
- **Yes** if using supervised outlier detection methods

**Default Paths Based on Model Type:**

| **Model Type**           | **Outlier Handling Techniques**                                     |
|--------------------------|----------------------------------------------------------------------|
| Linear/Logistic Regression | Z-Score filtering + IQR filtering                                |
| SVM & k-NN               | IQR filtering + Winsorization                                      |
| Neural Networks          | Winsorization/Clipping                                              |
| Clustering Algorithms    | IsolationForest or DBSCAN-based outlier detection                  |
| Tree-Based Models        | Typically no action needed                                         |

**Implementation Example:**

```python
from scipy import stats
from scipy.stats.mstats import winsorize
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
from typing import Tuple, Optional

class DataPreprocessor:
    # ... [Previous code]

    def handle_outliers(
        self, 
        X_train: pd.DataFrame, 
        y_train: Optional[pd.Series] = None, 
        debug: bool = False
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Handle outliers based on the model's sensitivity and user options.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (Optional[pd.Series]): Training target.
            debug (bool): Flag to control debug outputs.

        Returns:
            Tuple[pd.DataFrame, Optional[pd.Series]]: X_train without outliers and corresponding y_train.
        """
        step_name = "Handle Outliers"
        self.logger.info(f"Step: {step_name}")
        if debug:
            self.logger.debug("Starting outlier handling.")

        initial_shape = X_train.shape[0]
        new_columns = []

        # Fetch user-defined outlier handling options or set defaults
        outlier_options = self.options.get('handle_outliers', {})
        zscore_threshold = outlier_options.get('zscore_threshold', 3)
        iqr_multiplier = outlier_options.get('iqr_multiplier', 1.5)
        isolation_contamination = outlier_options.get('isolation_contamination', 0.05)

        # Check for target leakage: Ensure y_train is not used in transformations
        if self.mode == 'train' and y_train is not None and debug:
            self.logger.debug("y_train is present. Confirming it's not used in outlier handling.")

        for col in self.numericals:
            if self.model_category in ['regression', 'classification']:
                # Z-Score Filtering
                apply_zscore = outlier_options.get('apply_zscore', True)
                if apply_zscore:
                    z_scores = np.abs((X_train[col] - X_train[col].mean()) / X_train[col].std())
                    mask_z = z_scores < zscore_threshold
                    removed_z = (~mask_z).sum()
                    X_train = X_train[mask_z]
                    if y_train is not None:
                        y_train = y_train.loc[X_train.index]
                    self.feature_reasons[col] += f'Outliers handled with Z-Score Filtering (threshold={zscore_threshold}) | '
                    if debug:
                        self.logger.debug(f"Removed {removed_z} outliers from '{col}' using Z-Score Filtering")

                # IQR Filtering
                apply_iqr = outlier_options.get('apply_iqr', True)
                if apply_iqr:
                    Q1 = X_train[col].quantile(0.25)
                    Q3 = X_train[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - iqr_multiplier * IQR
                    upper_bound = Q3 + iqr_multiplier * IQR
                    mask_iqr = (X_train[col] >= lower_bound) & (X_train[col] <= upper_bound)
                    removed_iqr = (~mask_iqr).sum()
                    X_train = X_train[mask_iqr]
                    if y_train is not None:
                        y_train = y_train.loc[X_train.index]
                    self.feature_reasons[col] += f'Outliers handled with IQR Filtering (multiplier={iqr_multiplier}) | '
                    if debug:
                        self.logger.debug(f"Removed {removed_iqr} outliers from '{col}' using IQR Filtering")

            elif self.model_category == 'clustering':
                # IsolationForest
                iso_forest = IsolationForest(contamination=isolation_contamination, random_state=42)
                preds = iso_forest.fit_predict(X_train[[col]])
                mask_iso = preds != -1
                removed_iso = (preds == -1).sum()
                X_train = X_train[mask_iso]
                self.feature_reasons[col] += f'Outliers handled with IsolationForest (contamination={isolation_contamination}) | '
                if debug:
                    self.logger.debug(f"Removed {removed_iso} outliers from '{col}' using IsolationForest")

            elif self.model_category == 'tree_based_models':
                # Typically no outlier handling
                if debug:
                    self.logger.debug(f"No outlier handling applied for tree-based model '{self.model_type}'.")
                continue

            else:
                if debug:
                    self.logger.warning(f"Model category '{self.model_category}' not recognized for outlier handling.")

        self.preprocessing_steps.append("Handle Outliers")

        # Completion Logging
        if debug:
            self.logger.debug(f"Completed: Handle Outliers. Initial samples: {initial_shape}, Final samples: {X_train.shape[0]}")
            self.logger.debug(f"Missing values after outlier handling in X_train:\n{X_train.isnull().sum()}")
            self.logger.debug(f"Outlier handling applied on columns: {new_columns}")
        else:
            self.logger.info(f"Step '{step_name}' completed.")

        return X_train, y_train
```

**Options:**

- **Custom Detection Methods:**
  - Implement domain-specific outlier detection mechanisms tailored to the dataset characteristics.

- **Alternative Techniques:**
  - Use `RobustScaler` for scaling instead of removing outliers to reduce their impact without data loss.
  
    ```python
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X[numerical_features])
    ```
  
- **Model-Based Outlier Handling:**
  - Utilize models that are inherently robust to outliers (e.g., tree-based models, ensemble methods).

---

### 6. Choose and Apply Transformations (Based on Normality Tests)

**Goal:**
Apply transformations to achieve distributions closer to model assumptions.

**Requires y:**
- **No**

**Default Paths Based on Model Type:**

| **Model Type**            | **Transformation Strategy**                                                                                     |
|---------------------------|-----------------------------------------------------------------------------------------------------------------|
| Linear/Logistic Regression | Apply Box-Cox or Yeo-Johnson transform if p-value < 0.05 OR skewness > threshold                               |
| Neural Networks, SVM, k-NN, Clustering | Apply Yeo-Johnson transform if skewness > threshold                                                   |
| Tree-Based Models         | Only transform if skewness > threshold                                                                         |

**Implementation Example:**

```python
from sklearn.preprocessing import PowerTransformer
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

class DataPreprocessor:
    # ... [Previous code]

    def apply_transformations(
        self, 
        X_train: pd.DataFrame, 
        X_test: Optional[pd.DataFrame] = None, 
        features_to_transform: List[str] = [], 
        debug: bool = False
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Apply transformations to numerical features based on normality test results.

        Args:
            X_train (pd.DataFrame): Training features.
            X_test (Optional[pd.DataFrame]): Testing features.
            features_to_transform (List[str]): List of numerical features that need transformation.
            debug (bool): Flag to control debug outputs.

        Returns:
            Tuple[pd.DataFrame, Optional[pd.DataFrame]]: Transformed X_train and X_test.
        """
        step_name = "Choose and Apply Transformations"
        self.logger.info(f"Step: {step_name}")
        if debug:
            self.logger.debug(f"Features to transform: {features_to_transform}")

        if not features_to_transform:
            self.logger.info("No transformations required based on normality tests.")
            self.preprocessing_steps.append(step_name)
            return X_train, X_test

        # Initialize PowerTransformer
        pt = PowerTransformer(method='yeo-johnson')
        pt.fit(X_train[features_to_transform])
        X_train_transformed = X_train.copy()
        X_train_transformed[features_to_transform] = pt.transform(X_train[features_to_transform])

        if X_test is not None:
            X_test_transformed = X_test.copy()
            X_test_transformed[features_to_transform] = pt.transform(X_test[features_to_transform])
        else:
            X_test_transformed = None

        # Store the transformer for inverse transformations and prediction data
        self.transformers['power'] = pt

        self.preprocessing_steps.append(step_name)

        # Completion Logging
        if debug:
            self.logger.debug(f"Completed: {step_name}. Transformed features: {features_to_transform}")
            self.logger.debug(f"X_train transformed shape: {X_train_transformed.shape}")
            if X_test_transformed is not None:
                self.logger.debug(f"X_test transformed shape: {X_test_transformed.shape}")
        else:
            self.logger.info(f"Step '{step_name}' completed: Applied transformations to specified features.")

        return X_train_transformed, X_test_transformed
```

**Options:**

- **Multiple Transformations:**
  - Experiment with different transformation techniques (e.g., Box-Cox, Yeo-Johnson, log) to identify the best fit for skewed features.
  
- **Pipeline Integration:**
  - Incorporate transformations into a `Pipeline` or `ColumnTransformer` to streamline preprocessing.
  
    ```python
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    transformer = ColumnTransformer([
        ('power', PowerTransformer(), features_to_transform),
        # Add other transformers if needed
    ])

    transformer.fit(X_train)
    X_train_transformed = transformer.transform(X_train)
    X_test_transformed = transformer.transform(X_test)
    ```
  
- **Custom Transformations:**
  - Apply domain-specific transformations as required by the dataset or model.
  
    ```python
    import numpy as np

    # Example: Log Transformation
    X_train[features_to_transform] = np.log1p(X_train[features_to_transform])
    X_test[features_to_transform] = np.log1p(X_test[features_to_transform])
    ```

**Considerations for Prediction Preparation:**

- **Consistent Application:**
  - Use the same transformation parameters fitted on the training data to transform new data.
  
    ```python
    # Apply fitted PowerTransformer to new data
    X_new_imputed[features_to_transform] = self.transformers['power'].transform(
        X_new_imputed[features_to_transform]
    )
    ```
  
- **Handling Infeasible Transformations:**
  - If certain transformations are not feasible for new data (e.g., log transform on non-positive values), decide on a strategy to handle them (e.g., clipping or using alternative transformations).

---

### 7. Encode Categorical Variables

**Goal:**
Convert categorical data into numeric form, ensuring that categorical relationships and structures are preserved while making the data suitable for the chosen model.

**Requires y:**
- **No** for standard encoding
- **Yes** for supervised encoding techniques like target encoding

**Default Paths Based on Model Type:**

| **Model Type**           | **Encoding Strategy**                                                                                         |
|--------------------------|----------------------------------------------------------------------------------------------------------------|
| Linear/Logistic Regression | OrdinalEncoder for ordinal features; OneHotEncoder for nominal features.                                     |
| Neural Networks, SVM, k-NN, Clustering | OrdinalEncoder for ordinal features; OneHotEncoder for nominal features to avoid implying order where none exists. |
| Tree-Based Models         | OrdinalEncoder for both ordinal and nominal features, as trees can handle splits on ordinally encoded features effectively without one-hot encoding. |

**Handling SMOTENC with Ordinal Encoding for Nominal Categorical Features:**

When using SMOTE variants that handle categorical features (e.g., `SMOTENC`), nominal features should be ordinally encoded to reduce dimensionality and maintain compatibility.

**Step-by-Step Implementation:**

1. **Identify Nominal and Ordinal Categorical Features:**

    ```python
    # Example: Define nominal and ordinal features
    nominal_features = ['nominal_cat1', 'nominal_cat2']  # Replace with your nominal categorical feature names
    ordinal_features = ['ordinal_cat1', 'ordinal_cat2']  # Replace with your ordinal categorical feature names
    ```

2. **Apply Ordinal Encoding to Nominal Categorical Features:**

    ```python
    from sklearn.preprocessing import OrdinalEncoder

    # Initialize OrdinalEncoder
    ordinal_encoder_nominal = OrdinalEncoder()

    # Fit and transform nominal features
    X_train_imputed[nominal_features] = ordinal_encoder_nominal.fit_transform(
        X_train_imputed[nominal_features]
    )
    X_test_imputed[nominal_features] = ordinal_encoder_nominal.transform(
        X_test_imputed[nominal_features]
    )

    # Store the encoder for inverse transformations and prediction data
    self.encoders['ordinal_nominal'] = ordinal_encoder_nominal
    ```

    **Rationale:**
    - Reduces dimensionality compared to OneHotEncoder and is compatible with SMOTENC.

3. **Encode Ordinal Categorical Features:**

    ```python
    # Initialize OrdinalEncoder for ordinal features
    ordinal_encoder_ord = OrdinalEncoder()

    # Fit and transform ordinal features
    X_train_imputed[ordinal_features] = ordinal_encoder_ord.fit_transform(
        X_train_imputed[ordinal_features]
    )
    X_test_imputed[ordinal_features] = ordinal_encoder_ord.transform(
        X_test_imputed[ordinal_features]
    )

    # Store the encoder for inverse transformations and prediction data
    self.encoders['ordinal_ord'] = ordinal_encoder_ord
    ```

4. **Summary of Encoded Features:**

    ```python
    # Combine all encoded categorical features
    encoded_categorical_features = nominal_features + ordinal_features
    ```

**Options:**

- **OneHotEncoder for Nominal Features:**
  - Particularly for models sensitive to ordinal encoding (e.g., linear models), consider OneHotEncoder to avoid implying order.
  
    ```python
    from sklearn.preprocessing import OneHotEncoder

    # Initialize OneHotEncoder
    onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Fit and transform nominal features
    X_train_onehot = onehot_encoder.fit_transform(X_train_imputed[nominal_features])
    X_test_onehot = onehot_encoder.transform(X_test_imputed[nominal_features])

    # Get feature names after one-hot encoding
    onehot_feature_names = onehot_encoder.get_feature_names_out(nominal_features)

    # Convert to DataFrame
    X_train_onehot_df = pd.DataFrame(X_train_onehot, columns=onehot_feature_names, index=X_train_imputed.index)
    X_test_onehot_df = pd.DataFrame(X_test_onehot, columns=onehot_feature_names, index=X_test_imputed.index)

    # Drop original nominal features and concatenate one-hot encoded features
    X_train_encoded = X_train_imputed.drop(columns=nominal_features).join(X_train_onehot_df)
    X_test_encoded = X_test_imputed.drop(columns=nominal_features).join(X_test_onehot_df)

    # Store OneHotEncoder for inverse transformations and prediction data
    self.encoders['onehot'] = onehot_encoder

    # Update encoded_categorical_features
    encoded_categorical_features = onehot_feature_names.tolist() + ordinal_features
    ```
  
- **Target Encoding for High Cardinality Nominal Features:**
  - For high cardinality nominal features, use target encoding techniques that may leverage y.
  
    ```python
    import category_encoders as ce

    target_encoder = ce.TargetEncoder(cols=nominal_features)
    X_train_imputed[nominal_features] = target_encoder.fit_transform(X_train_imputed[nominal_features], y_train)
    X_test_imputed[nominal_features] = target_encoder.transform(X_test_imputed[nominal_features])

    # Store TargetEncoder for inverse transformations and prediction data
    self.encoders['target'] = target_encoder
    ```

**Complete Encoding Example:**

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import pandas as pd

# Define nominal and ordinal features
nominal_features = ['nominal_cat1', 'nominal_cat2']  # Replace with actual names
ordinal_features = ['ordinal_cat1', 'ordinal_cat2']  # Replace with actual names

# Ordinal Encoding for nominal features
ordinal_encoder_nominal = OrdinalEncoder()
X_train_imputed[nominal_features] = ordinal_encoder_nominal.fit_transform(
    X_train_imputed[nominal_features]
)
X_test_imputed[nominal_features] = ordinal_encoder_nominal.transform(
    X_test_imputed[nominal_features]
)

# Ordinal Encoding for ordinal features
ordinal_encoder_ord = OrdinalEncoder()
X_train_imputed[ordinal_features] = ordinal_encoder_ord.fit_transform(
    X_train_imputed[ordinal_features]
)
X_test_imputed[ordinal_features] = ordinal_encoder_ord.transform(
    X_test_imputed[ordinal_features]
)

# Combine all encoded categorical features
encoded_categorical_features = nominal_features + ordinal_features

# Store encoders for future use
self.encoders['ordinal_nominal'] = ordinal_encoder_nominal
self.encoders['ordinal_ord'] = ordinal_encoder_ord
```

**Considerations for Prediction Preparation:**

- **Use Fitted Encoders:**
  - Apply the same encoders fitted on training data to transform new data.
  
    ```python
    # Apply fitted OrdinalEncoder to new data
    X_new_imputed[nominal_features] = self.encoders['ordinal_nominal'].transform(
        X_new_imputed[nominal_features]
    )
    X_new_imputed[ordinal_features] = self.encoders['ordinal_ord'].transform(
        X_new_imputed[ordinal_features]
    )
    ```
  
- **Handle Unseen Categories:**
  - Ensure encoders are set to handle unknown categories gracefully to prevent errors during transformation.
  
    ```python
    onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ```

---

### 8. Apply Scaling (If Needed by Model)

**Goal:**
Normalize feature scales so that features contribute appropriately, especially in distance or gradient-based models.

**Requires y:**
- **No**

**Default Paths Based on Model Type:**

| **Model Type**              | **Scaling Strategy**                                                                 |
|-----------------------------|---------------------------------------------------------------------------------------|
| Linear Regression & Logistic Regression | Use `StandardScaler` to center data at zero mean and unit variance.                  |
| Neural Networks, SVM, k-NN, Clustering | Use `MinMaxScaler` for distance-based computations to ensure all features are on a comparable scale. |
| Tree-Based Models           | No scaling needed, as trees are scale-invariant.                                     |

**Implementation Example:**

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pandas as pd
from typing import Tuple, Optional

class DataPreprocessor:
    # ... [Previous code]

    def apply_scaling(
        self, 
        X_train: pd.DataFrame, 
        X_test: Optional[pd.DataFrame] = None, 
        debug: bool = False
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Apply scaling to numerical features based on model requirements.

        Args:
            X_train (pd.DataFrame): Training features.
            X_test (Optional[pd.DataFrame]): Testing features.
            debug (bool): Flag to control debug outputs.

        Returns:
            Tuple[pd.DataFrame, Optional[pd.DataFrame]]: Scaled X_train and X_test.
        """
        step_name = "Apply Scaling"
        self.logger.info(f"Step: {step_name}")
        numerical_features = self.numericals  # Assuming self.numericals contains numerical feature names

        # Determine scaling strategy based on model type
        scaling_strategy = None
        if self.model_category in ['regression', 'classification']:
            if self.model_type.lower() in ['linear regression', 'logistic regression']:
                scaling_strategy = 'StandardScaler'
            elif self.model_type.lower() in ['neural networks', 'svm', 'k_nn', 'clustering']:
                scaling_strategy = 'MinMaxScaler'
        elif self.model_category == 'tree_based_models':
            scaling_strategy = None  # No scaling needed
        else:
            scaling_strategy = None  # Default to no scaling

        if scaling_strategy == 'StandardScaler':
            scaler = StandardScaler()
        elif scaling_strategy == 'MinMaxScaler':
            scaler = MinMaxScaler()
        elif scaling_strategy == 'RobustScaler':
            scaler = RobustScaler()
        else:
            scaler = None

        if scaler:
            scaler.fit(X_train[numerical_features])
            X_train_scaled = X_train.copy()
            X_train_scaled[numerical_features] = scaler.transform(X_train[numerical_features])

            if X_test is not None:
                X_test_scaled = X_test.copy()
                X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])
            else:
                X_test_scaled = None

            # Store scaler for inverse transformations and prediction data
            self.scalers['scaler'] = scaler

            self.preprocessing_steps.append(step_name)

            # Completion Logging
            if debug:
                self.logger.debug(f"Completed: {step_name}. Applied {scaling_strategy} to numerical features.")
                self.logger.debug(f"X_train_scaled shape: {X_train_scaled.shape}")
                if X_test_scaled is not None:
                    self.logger.debug(f"X_test_scaled shape: {X_test_scaled.shape}")
            else:
                self.logger.info(f"Step '{step_name}' completed: Applied {scaling_strategy} to numerical features.")

            return X_train_scaled, X_test_scaled
        else:
            self.logger.info("No scaling applied based on model type.")
            self.preprocessing_steps.append(step_name)
            return X_train, X_test
```

**Options:**

- **Alternative Scalers:**
  - Use `RobustScaler` if data contains outliers to reduce their impact.
  
    ```python
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X[numerical_features])
    ```
  
- **Model-Specific Scaling:**
  - For SVM/k-NN, experiment with `StandardScaler` instead of `MinMaxScaler` to see if performance improves.
  
- **No Scaling for Tree-Based Models:**
  - Set `scaler = None` and skip scaling steps.
  
    ```python
    scaler = None
    ```

**Handling Scaling in the Pipeline:**

```python
# Example: Using MinMaxScaler for SVM/k-NN
scaler = MinMaxScaler()

# Example: No scaling for tree-based models
scaler = None

if scaler is not None:
    X_train_scaled = scaler.fit_transform(X_train_final[numerical_features])
    X_test_scaled = scaler.transform(X_test_final[numerical_features])
    
    # Update the DataFrames with scaled numerical features
    X_train_final[numerical_features] = X_train_scaled
    X_test_final[numerical_features] = X_test_scaled
else:
    X_train_scaled = X_train_final
    X_test_scaled = X_test_final
```

**Considerations for Prediction Preparation:**

- **Consistent Application:**
  - Use the same scaler fitted on the training data to transform new data.
  
    ```python
    # Apply fitted scaler to new data
    X_new_scaled = self.scalers['scaler'].transform(X_new_final[numerical_features])
    
    # Reconstruct DataFrame
    X_new_final[numerical_features] = X_new_scaled
    ```
  
- **Handling New Features:**
  - Ensure that new data contains the same numerical features as the training data.

---

### 9. Implement SMOTE (Train Only)

**Goal:**
Address class imbalance in classification tasks by generating synthetic minority class samples.

**Requires y:**
- **Yes**, as SMOTE is a supervised technique.

**Default Paths Based on Model Type:**

| **Model Type**          | **SMOTE Strategy**                                                                                                             |
|-------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| Classification Tasks with only Numerical Datasets | Choose based on imbalance severity and data characteristics: <br> - Severe imbalance: Use `ADASYN` <br> - Noisy data: Use `SMOTEENN` <br> - High overlap: Use `SMOTETomek` <br> - Boundary concentration: Use `BorderlineSMOTE` <br> - Normal imbalance: Use `SMOTE` |
| Models with Categorical Data | Use `SMOTENC` if mixed numerical/categorical. <br> Use `SMOTEN` if only categorical.                                     |
| Regression or Non-Classification Tasks | **Do not apply SMOTE**                                                                                                    |

**Adjustments for SMOTENC with Ordinal Encoding for Nominal Categorical Features:**

To integrate `SMOTENC` with ordinal encoding for nominal categorical features, follow these steps only for classification models that involve mixed numerical and categorical features.

**Step-by-Step Implementation:**

1. **Import SMOTENC and OrdinalEncoder:**

    ```python
    from imblearn.over_sampling import SMOTENC
    from sklearn.preprocessing import OrdinalEncoder
    ```

2. **Define Categorical Features for SMOTENC:**

    ```python
    # Combine nominal and ordinal features
    categorical_features_smote = nominal_features + ordinal_features

    # Get the indices of categorical features
    categorical_feature_indices = [
        X_train_final.columns.get_loc(col) for col in categorical_features_smote
    ]
    ```

3. **Initialize SMOTENC with OrdinalEncoder:**

    ```python
    smote = SMOTENC(
        categorical_features=categorical_feature_indices,
        categorical_encoder=OrdinalEncoder(),
        sampling_strategy='auto',
        random_state=42,
        k_neighbors=5
    )
    ```

4. **Fit and Resample the Training Data:**

    ```python
    X_train_res, y_train_res = smote.fit_resample(X_train_final, y_train_filtered)
    ```

5. **Store SMOTENC Instance for Future Use:**

    ```python
    self.smote = smote
    ```

**Options:**

- **Adjust `sampling_strategy`:**
  - Depending on the severity of class imbalance, adjust the strategy to control the extent of oversampling.
  
    ```python
    smote = SMOTE(sampling_strategy=0.5, random_state=self.random_state)  # Minority class will be 50% the size of the majority class
    ```
  
- **Modify `k_neighbors`:**
  - Control the neighborhood size for synthetic sample generation.
  
    ```python
    smote = SMOTE(k_neighbors=3, random_state=self.random_state)
    ```
  
- **Choose a Different SMOTE Variant:**
  - Select a variant that best fits the data characteristics.
  
    ```python
    from imblearn.over_sampling import BorderlineSMOTE

    smote = BorderlineSMOTE(sampling_strategy='auto', random_state=self.random_state)
    ```

**Important Considerations:**

- **Ordinal Encoding of Nominal Features:**
  - Imposes an arbitrary order. Ensure that your model can handle this without misinterpretation. Tree-based models are generally robust to such encodings, but other models might interpret the encoded values as having inherent order.
  
- **Inverse Transformation:**
  - Keep track of the encoders used within `SMOTENC` for later inverse transformations if needed.

**Implementation Example:**

```python
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import OrdinalEncoder

# Combine nominal and ordinal features
categorical_features_smote = nominal_features + ordinal_features

# Get the indices of categorical features
categorical_feature_indices = [
    X_train_final.columns.get_loc(col) for col in categorical_features_smote
]

# Initialize SMOTENC with OrdinalEncoder
smote = SMOTENC(
    categorical_features=categorical_feature_indices,
    categorical_encoder=OrdinalEncoder(),
    sampling_strategy='auto',
    random_state=42,
    k_neighbors=5
)

# Fit and resample
X_train_res, y_train_res = smote.fit_resample(X_train_final, y_train_filtered)

# Store SMOTENC instance
self.smote = smote
```

**Considerations for Prediction Preparation:**

- **Do Not Apply SMOTE to New Data:**
  - SMOTE is a training technique to balance classes and should not be applied to new or unseen data intended for prediction.

---

### 10. Train Model on Preprocessed Training Data

**Goal:**
Fit the chosen model to the fully preprocessed, balanced training data.

**Requires y:**
- **Yes**

**Default Paths Based on Model Type:**

| **Model Type**             | **Training Requirements**                                                                              |
|----------------------------|--------------------------------------------------------------------------------------------------------|
| Linear/Logistic Regression | Train with scaled and encoded data.                                                                   |
| Neural Networks, SVM, k-NN, Clustering | Train with scaled and encoded data.                                                       |
| Tree-Based Models          | Train with encoded (ordinal by default) but unscaled data.                                           |

**Options:**

- **Regularization:**
  - For regression models, consider ridge or lasso regularization to prevent overfitting.
  
    ```python
    from sklearn.linear_model import Ridge, Lasso

    # Ridge Regression
    model = Ridge(alpha=1.0, random_state=42)

    # Lasso Regression
    model = Lasso(alpha=0.1, random_state=42)
    ```
  
- **Model Selection:**
  - Experiment with different classifiers (e.g., `RandomForestClassifier` instead of `LogisticRegression`).
  
    ```python
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    ```
  
- **Hyperparameter Tuning:**
  - Utilize grid search or randomized search to find optimal hyperparameters.
  
    ```python
    from sklearn.model_selection import GridSearchCV

    param_grid = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
    grid = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5)
    grid.fit(X_train_res, y_train_res)
    best_model = grid.best_estimator_
    ```

**Implementation Example:**

```python
from sklearn.linear_model import LogisticRegression

# Initialize the model
model = LogisticRegression(random_state=42)

# Fit the model
model.fit(X_train_res, y_train_res)

# OPTION: Switch to another model
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(random_state=42)
# model.fit(X_train_res, y_train_res)
```

**Additional Notes:**

- **Model Selection:**
  - Choose models based on the problem type, data characteristics, and performance metrics.
  
- **Regularization:**
  - Helps in preventing overfitting, especially for models like Logistic Regression.

---

### 11. Predict on Test Data (No SMOTE on Test)

**Goal:**
Evaluate model performance on the original, untouched test set, ensuring a real-world performance estimate.

**Requires y:**
- **Yes**, for evaluating performance.

**Default Paths:**

- **Apply the same transformations (encoding, scaling) used on training data to the test set.**
- **Do not apply SMOTE to the test set.**

**Options:**

- **Avoid Synthetic Augmentation:**
  - Applying SMOTE-like methods on the test set is not recommended as it distorts the true distribution.
  
- **Separate Evaluation Scenarios:**
  - If needed, create a separate evaluation scenario with SMOTE-like methods, but primarily, keep the test set untouched.

**Implementation Example:**

```python
# Predict
y_pred = model.predict(X_test_scaled)

# Evaluate
from sklearn.metrics import accuracy_score, f1_score

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

# OPTION: Evaluate with additional metrics
# from sklearn.metrics import roc_auc_score, precision_score, recall_score
# print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]))
# print("Precision:", precision_score(y_test, y_pred, average='weighted'))
# print("Recall:", recall_score(y_test, y_pred, average='weighted'))
```

**Additional Metrics (OPTION):**

- **ROC-AUC:**
  - For binary or multi-class classification, provides insight into the model's ability to discriminate between classes.
  
    ```python
    from sklearn.metrics import roc_auc_score, precision_score, recall_score

    print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    ```
  
- **Confusion Matrix:**
  - To visualize true vs. predicted classes.
  
    ```python
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    ```

---

### 12. Final Inverse Transformations for Interpretability

**Goal:**
Revert preprocessed data (scaled, encoded, transformed) back to its original form for interpretability and reporting.

**Requires y:**
- **No**, as it pertains to transforming feature data.

**Default Paths:**

- **Inverse Scaling:**
  - Using the scaler’s `inverse_transform` if scaling was applied.
  
- **Inverse Encoding:**
  - **Ordinal:** Use `OrdinalEncoder`’s `inverse_transform`.
  - **One-Hot:** Use `OneHotEncoder`’s `inverse_transform` to reconstruct original categories.
  
- **Inverse Transformations for Log or Power Transforms:**
  - Apply the inverse (e.g., exponential for log, inverse transform for `PowerTransformer`).

**Options:**

- **Multiple Transformations:**
  - If multiple transformations were tested, track and apply the correct inverse.
  
- **Embeddings (NN-specific):**
  - Retain mapping for interpretability or skip if direct inverse is not feasible.

**Implementation Example:**

```python
# Inverse scaling
if 'scaler' in self.scalers:
    X_test_original_scale = X_test_scaled.copy()
    X_test_original_scale[numerical_features] = self.scalers['scaler'].inverse_transform(
        X_test_scaled[numerical_features]
    )
else:
    X_test_original_scale = X_test_scaled.copy()

# Inverse encoding for nominal features (OrdinalEncoder used instead of OneHotEncoder)
if 'ordinal_nominal' in self.encoders:
    X_test_original_scale[nominal_features] = self.encoders['ordinal_nominal'].inverse_transform(
        X_test_scaled[nominal_features]
    )

# Inverse encoding for ordinal features
if 'ordinal_ord' in self.encoders:
    X_test_original_scale[ordinal_features] = self.encoders['ordinal_ord'].inverse_transform(
        X_test_scaled[ordinal_features]
    )

# If PowerTransformer was used, apply inverse transform
if 'power' in self.transformers and features_to_transform:
    X_test_original_scale[features_to_transform] = self.transformers['power'].inverse_transform(
        X_test_original_scale[features_to_transform]
    )

# Combine to reconstruct full original data
# Assuming no other transformations were applied
```

**Important Considerations:**

- **Track Transformers:**
  - Ensure that all transformers (`scalers`, `encoders`, `power transformers`) used during preprocessing are saved and accessible for inverse transformations.
  
- **Consistent Order:**
  - Apply inverse transformations in the reverse order of the preprocessing steps to maintain data integrity.

**Considerations for Prediction Preparation:**

- **Inverse Transformations Typically Not Required:**
  - When preparing data solely for prediction, inverse transformations are generally unnecessary unless for specific interpretability purposes post-prediction.

---

### 13. Final Inverse Transformation Validation

**Goal:**
Validate that the inverse transformations restore the data to its near-original form, ensuring interpretability is accurate.

**Requires y:**
- **No**

**Default Paths:**

- **Numerical Features:**
  - Compute Mean Absolute Error (MAE) or similar metric between original and inverse-transformed values.
  
- **Categorical Features:**
  - Verify exact matches between original and inverse-transformed categories.

**Options:**

- **Adjust Tolerance:**
  - Modify tolerance for numerical differences based on precision requirements.
  
- **Perform Visual Checks:**
  - Plot distributions of original vs. inverse-transformed features.
  
    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import probplot

    # Numerical Features
    for col in numerical_features:
        plt.figure(figsize=(10, 4))
        sns.kdeplot(X_test[col], label='Original', shade=True)
        sns.kdeplot(X_test_original_scale[col], label='Inverse Transformed', shade=True)
        plt.title(f'Distribution Comparison for {col}')
        plt.legend()
        plt.show()
    ```
  
- **Statistical Tests:**
  - Perform statistical tests to assess the similarity between original and transformed data distributions.
  
    ```python
    from scipy.stats import ttest_ind

    # For numerical features
    for col in numerical_features:
        stat, p_val = ttest_ind(X_test[col].dropna(), X_test_original_scale[col].dropna())
        print(f"T-Test for {col}: stat={stat}, p-value={p_val}")
    ```

**Implementation Example:**

```python
import numpy as np

# Assume `X_test_original` is the original test data before preprocessing
# and `X_test_original_scale` is after inverse transformations

# For numerical features
diff = np.abs(X_test[numerical_features] - X_test_original_scale[numerical_features])
mae = diff.mean().mean()
print("Mean Absolute Error (MAE) on numerical features:", mae)

# For categorical features
categorical_features = nominal_features + ordinal_features
categorical_match = (
    X_test[categorical_features].astype(str) == 
    X_test_original_scale[categorical_features].astype(str)
)

if categorical_match.all().all():
    print("Categorical features match after inverse transformation.")
else:
    mismatches = categorical_match.apply(lambda x: not x.all(), axis=1)
    print(f"Found {mismatches.sum()} mismatched samples in categorical features.")
```

**Additional Validation Techniques:**

- **Visualization:**
  - Use histograms, box plots, or scatter plots to visually compare original and inverse-transformed data.
  
    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Numerical Features
    for col in numerical_features:
        plt.figure(figsize=(10, 4))
        sns.kdeplot(X_test[col], label='Original', shade=True)
        sns.kdeplot(X_test_original_scale[col], label='Inverse Transformed', shade=True)
        plt.title(f'Distribution Comparison for {col}')
        plt.legend()
        plt.show()
    ```
  
- **Statistical Tests:**
  - Perform statistical tests to assess the similarity between original and transformed data distributions.
  
    ```python
    from scipy.stats import ttest_ind

    # For numerical features
    for col in numerical_features:
        stat, p_val = ttest_ind(X_test[col].dropna(), X_test_original_scale[col].dropna())
        print(f"T-Test for {col}: stat={stat}, p-value={p_val}")
    ```

**Considerations for Prediction Preparation:**

- **Inverse Transformations Not Typically Needed:**
  - Since prediction preparation focuses on transforming data for model input, inverse transformations are generally unnecessary unless for specific interpretability tasks post-prediction.

---

## Automated Output Datasets

To facilitate automation and ensure consistency across different modes (`train`, `predict`, `clustering`), the pipeline outputs specific datasets tailored to each scenario. Below is a summary of the output datasets per mode:

### Training Mode

- **Outputs:**
  - `X_train_preprocessed`: Preprocessed and scaled training features.
  - `X_test_preprocessed`: Preprocessed and scaled testing features.
  - `y_train`: Training target variable.
  - `y_test`: Testing target variable.
  - `recommendations`: Preprocessing recommendations based on data analysis.
  - `X_train_inversed`: Inverse-transformed training features for interpretability (optional).
  - `X_test_inversed`: Inverse-transformed testing features for interpretability (optional).

### Prediction Mode

- **Outputs:**
  - `X_new_preprocessed`: Preprocessed and scaled new/unseen features ready for prediction.
  - `recommendations`: Preprocessing recommendations based on new data analysis.
  - `X_new_inversed`: Inverse-transformed new features for interpretability (optional).

### Clustering Mode

- **Outputs:**
  - `X_preprocessed`: Preprocessed and scaled features tailored for clustering algorithms.
  - `recommendations`: Preprocessing recommendations based on clustering data analysis.
  - `X_inversed`: Inverse-transformed features for interpretability (optional).

**Note:** The inverse-transformed datasets (`X_inversed`) are optional and primarily used for interpretability purposes. They allow you to revert the preprocessed data back to its original form, aiding in understanding and reporting.

---

## SMOTE Calculations for Numericals

Incorporating SMOTE recommendations based on class distribution and data characteristics ensures that you choose the most appropriate oversampling technique. Below is a detailed approach to calculate and recommend SMOTE variants:

**Step-by-Step Implementation:**

1. **Step 1: Class Distribution**

    ```python
    # Step 1: Class Distribution
    class_distribution = y_train.value_counts(normalize=True)
    majority_class = class_distribution.idxmax()
    minority_class = class_distribution.idxmin()

    severe_imbalance = class_distribution[minority_class] < self.imbalance_threshold
    extreme_imbalance = class_distribution[minority_class] < self.options.get('smote_recommendation', {}).get('extreme_imbalance_threshold', 0.05)

    if debug:
        self.logger.debug(f"X_train Shape: {X_train.shape}")
        self.logger.debug(f"Class Distribution: {class_distribution.to_dict()}")
        if extreme_imbalance:
            self.logger.warning(f"Extreme imbalance detected: {class_distribution[minority_class]:.2%}")
    ```

2. **Step 2: Noise Analysis**

    ```python
    # Step 2: Noise Analysis
    minority_samples = X_train[y_train == minority_class]
    majority_samples = X_train[y_train == majority_class]

    try:
        knn = NearestNeighbors(n_neighbors=5).fit(majority_samples)
        distances, _ = knn.kneighbors(minority_samples)
        median_distance = np.median(distances)
        noise_ratio = np.mean(distances < median_distance)
        noisy_data = noise_ratio > self.noise_threshold

        if debug:
            self.logger.debug(f"Median Distance to Nearest Neighbors: {median_distance}")
            self.logger.debug(f"Noise Ratio: {noise_ratio:.2%}")
    except ValueError as e:
        self.logger.error(f"Noise analysis error: {e}")
        noisy_data = False
    ```

3. **Step 3: Overlap Analysis**

    ```python
    # Step 3: Overlap Analysis
    try:
        pdistances = pairwise_distances(minority_samples, majority_samples)
        overlap_metric = np.mean(pdistances < 1.0)
        overlapping_classes = overlap_metric > self.overlap_threshold

        if debug:
            self.logger.debug(f"Overlap Metric: {overlap_metric:.2%}")
    except ValueError as e:
        self.logger.error(f"Overlap analysis error: {e}")
        overlapping_classes = False
    ```

4. **Step 4: Boundary Concentration**

    ```python
    # Step 4: Boundary Concentration
    try:
        boundary_ratio = np.mean(np.min(distances, axis=1) < np.percentile(distances, 25))
        boundary_concentration = boundary_ratio > self.boundary_threshold

        if debug:
            self.logger.debug(f"Boundary Concentration Ratio: {boundary_ratio:.2%}")
    except Exception as e:
        self.logger.error(f"Boundary concentration error: {e}")
        boundary_concentration = False
    ```

5. **Step 5: Recommendations**

    ```python
    # Step 5: Recommendations
    recommendations = []
    if severe_imbalance:
        recommendations.append("ADASYN" if not noisy_data else "SMOTEENN")
    if noisy_data:
        recommendations.append("SMOTEENN")
    if overlapping_classes:
        recommendations.append("SMOTETomek")
    if boundary_concentration:
        recommendations.append("BorderlineSMOTE")
    if not recommendations:
        recommendations.append("SMOTE")

    if debug:
        self.logger.debug(f"SMOTE Recommendations: {recommendations}")
    ```

**Summary:**

Based on the calculations, the pipeline recommends appropriate SMOTE variants to address class imbalance effectively.

---

## Final Verification and Readiness for K-Means Clustering

After completing all preprocessing steps, ensure the dataset is ready for K-Means Clustering by verifying the following:

1. **Dataset Loading:**
   - Successfully loaded without issues.

2. **Feature Selection:**
   - Relevant features are selected and filtered correctly.

3. **Missing Values:**
   - All missing values have been imputed appropriately.

4. **Outlier Handling:**
   - Outliers have been effectively removed or handled, resulting in a cleaner dataset.

5. **Transformations:**
   - Numerical features are normalized using appropriate transformations, ensuring that K-Means isn't skewed by feature scales.

6. **Categorical Encoding:**
   - Categorical variables have been encoded numerically, converting them into formats suitable for K-Means clustering.

7. **Scaling:**
   - Numerical features have been scaled using `MinMaxScaler`, standardizing feature ranges, which is crucial for distance-based algorithms like K-Means.

8. **SMOTE Implementation:**
   - **Not Applied for K-Means**, as SMOTE is a supervised technique for handling class imbalance, which is irrelevant in unsupervised clustering.

9. **Model Training:**
   - K-Means has been trained on the fully preprocessed and appropriately scaled data.

10. **Prediction and Evaluation:**
    - Clustering results have been evaluated without applying SMOTE, ensuring that the true data distribution is maintained.

11. **Inverse Transformations:**
    - Successfully applied and validated, ensuring that preprocessing steps are reversible and accurate.

12. **Data Integrity:**
    - The final dataset is clean, scaled, and encoded, making it suitable for effective K-Means clustering.

**Final Dataset Characteristics for K-Means:**

- **Shape:** `(106, 15)` indicating 106 observations with 15 features.
- **Feature Types:** All features are numerical, suitable for K-Means.
- **Data Quality:** No missing values or outliers; features are scaled and encoded appropriately.
- **Integrity:** Inverse transformations validate the accuracy of preprocessing steps.

**Conclusion:**

Your data preprocessing pipeline has successfully prepared the dataset for K-Means Clustering. The data is clean, scaled, and encoded, ensuring that the clustering algorithm can effectively identify patterns and groupings within the data.

---

# **Usage Example**

Here's a brief example of how to utilize the `DataPreprocessor` class within your workflow:

```python
import pandas as pd

# Define column assets
column_assets = {
    'numericals': ['age', 'income', 'balance'],
    'ordinal_categoricals': ['education_level'],
    'nominal_categoricals': ['gender', 'marital_status'],
    'y_variable': ['target']
}

# Initialize DataPreprocessor
preprocessor = DataPreprocessor(
    model_type='Logistic Regression',
    column_assets=column_assets,
    mode='train',
    options={
        'handle_missing_values': {
            'numerical_strategy': {'strategy': 'mean', 'imputer': 'SimpleImputer'},
            'categorical_strategy': {'strategy': 'most_frequent', 'imputer': 'SimpleImputer'}
        },
        'split_dataset': {
            'test_size': 0.2,
            'random_state': 42,
            'stratify_for_classification': True
        },
        'handle_outliers': {
            'zscore_threshold': 3,
            'iqr_multiplier': 1.5,
            'apply_zscore': True,
            'apply_iqr': True
        },
        'test_normality': {
            'p_value_threshold': 0.05,
            'skewness_threshold': 1.0,
            'use_p_value_other_models': False
        },
        'smote_recommendation': {
            'imbalance_threshold': 0.1,
            'noise_threshold': 0.1,
            'overlap_threshold': 0.1,
            'boundary_threshold': 0.1,
            'extreme_imbalance_threshold': 0.05
        }
    },
    debug=True
)

# Load data
data = pd.read_csv('data.csv')
X = data.drop('target', axis=1)
y = data['target']

# Step 1: Handle Missing Values
X_train, X_test = preprocessor.handle_missing_values(X, y, debug=True)

# Step 2: Split Dataset
X_train, X_test, y_train, y_test = preprocessor.split_dataset(X_train, y, debug=True)

# Step 3: Handle Outliers
X_train, y_train = preprocessor.handle_outliers(X_train, y_train, debug=True)

# Step 4: Test for Normality
normality_results = preprocessor.test_normality(X_train, debug=True)

# Step 5: Apply Transformations
features_to_transform = [col for col, res in normality_results.items() if res['needs_transform']]
X_train, X_test = preprocessor.apply_transformations(X_train, X_test, features_to_transform, debug=True)

# Step 6: Encode Categorical Variables
X_train, X_test = preprocessor.encode_categorical_variables(X_train, X_test, debug=True)

# Step 7: Apply Scaling
X_train, X_test = preprocessor.apply_scaling(X_train, X_test, debug=True)

# Step 8: Implement SMOTE
X_train_res, y_train_res = preprocessor.implement_smote(X_train, y_train, debug=True)

# Step 9: Train Model
preprocessor.train_model(X_train_res, y_train_res, debug=True)

# Step 10: Predict on Test Data
preprocessor.predict_and_evaluate(X_test, y_test, debug=True)

# Step 11: Inverse Transformations (Optional)
X_test_original = preprocessor.inverse_transform(X_test, debug=True)

# Step 12: Validate Inverse Transformations
preprocessor.validate_inverse_transformations(X_test, X_test_original, debug=True)
```

---

## **Contribution Guidelines**

We welcome contributions to enhance the `DataPreprocessor` pipeline. Please follow these steps to contribute:

1. **Fork the Repository**
2. **Create a Feature Branch**
    ```bash
    git checkout -b feature/your-feature-name
    ```
3. **Commit Your Changes**
    ```bash
    git commit -m "Add your message"
    ```
4. **Push to the Branch**
    ```bash
    git push origin feature/your-feature-name
    ```
5. **Open a Pull Request**

Please ensure your code adheres to the existing coding standards and includes necessary tests.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **Contact**

For any inquiries or support, please open an issue on the repository or contact [your-email@example.com](mailto:ghadfiield32@gmail.com).

