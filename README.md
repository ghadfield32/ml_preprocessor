Below is a step‐by‐step explanation of how we merged the content from the two README files and incorporated the additional large table along with its contextual information. Following that is the final, combined README in markdown format.

---

## Step-by-Step Explanation

1. **Determine the Unified Title & Overview**  
   We started by consolidating the titles and opening paragraphs. The original READMEs are titled “DataPreprocessor for Time Series” while the additional content uses “DataPreprocessor.” Since the package covers both time series and general ML preprocessing, we chose a unified title that emphasizes the comprehensive nature of the tool. The overview now describes the package as a comprehensive solution for data preprocessing, highlighting both general functionalities (e.g., missing values, scaling, encoding, SMOTE) and specialized time series modes.

2. **Merge Features & Table of Contents**  
   - The “Features” section from the large table README was retained verbatim to emphasize key functionalities.
   - The Table of Contents was restructured to include all major sections from both inputs. This includes installation, usage examples (basic train, predict, and cluster modes), time series modes, splitting methods, the detailed “Preprocessing Pipeline Table,” evaluation & debugging, advanced usage, contributing guidelines, license, continuous integration, and resources.

3. **Combine Installation & Usage Sections**  
   - Installation instructions are nearly identical in both versions. We merged these without change.
   - For the Usage section, we combined examples from both sources: basic train examples, prediction mode, and even a cluster mode example. This ensures that users see multiple application examples.

4. **Incorporate Time Series Specific Sections**  
   - The “Time Series Modes” (set_window, pad, dtw) and “Splitting Methods” (percentage, date, sequence-aware, feature-engine) sections were kept intact.
   - An “Advanced Usage Example” for training an LSTM model with time series data is included as well.

5. **Embed the Large Table and Its Context**  
   - The comprehensive table that details the default pipeline configurations for each model type is added as a dedicated section titled “Detailed Preprocessing Pipeline Table.”
   - Accompanying explanations (columns summary, how to use the table, and customization tips) were also incorporated directly below the table.

6. **Concluding Sections**  
   - The Evaluation & Debugging, Contributing, License, Continuous Integration, and Resources sections are included from the larger README file to give users additional insights and guidance.

---

## Final Combined README.md

```markdown
# DataPreprocessor

**DataPreprocessor** is a comprehensive Python package designed to streamline the data preprocessing workflow for various machine learning models. It supports both general preprocessing tasks—such as handling missing values, outlier detection, encoding, scaling, and class imbalance handling—and advanced **time series modes** (`set_window`, `pad`, `dtw`) along with multiple **splitting methods** (`percentage-based`, `date-based`, `sequence-aware`, and PSI-based approaches).

> **Note:** Ensure you have performed any necessary feature engineering or multicollinearity checks before running DataPreprocessor.

---

## 🧰 Features

- **Flexible Preprocessing Pipeline:** Customize preprocessing steps based on your model type and mode. The default path supports most Regressive, Classifier, and Clustering models—with adjustments available as needed.
- **Automatic Outlier Detection:** Utilize methods like Z-Score Filtering, IQR Filtering, and Isolation Forest.
- **Categorical Encoding:** Support for both Ordinal and One-Hot Encoding.
- **Scaling Options:** Choose from StandardScaler, MinMaxScaler, or RobustScaler.
- **Class Imbalance Handling:** Automated handling through SMOTE variants:
  - For mixed numerical and categorical datasets, SMOTENC is applied.
  - For categorical-only datasets, SMOTEN is used.
  - For numerical-only datasets, a criteria-based approach using SMOTE, BorderlineSMOTE, ADASYN, SMOTEENN, or SMOTETomek is applied.
- **Configuration Driven:** Easily adjust steps via YAML configuration files.
- **Inverse Transformation:** Reconstruct original feature values from transformed data for interpretability.

---

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Basic Train Example](#basic-train-example)
  - [Preprocessing Configuration](#preprocessing-configuration)
  - [Running Preprocessing in Predict Mode](#running-preprocessing-in-predict-mode)
  - [Running Preprocessing in Cluster Mode](#running-preprocessing-in-cluster-mode)
- [Time Series Modes](#time-series-modes)
  - [set_window](#set_window)
  - [pad](#pad)
  - [dtw](#dtw)
- [Splitting Methods](#splitting-methods)
  - [Percentage-based Split](#percentage-based-split)
  - [Date-based Split](#date-based-split)
  - [Sequence-aware Split](#sequence-aware-split)
  - [Feature-Engine or PSI-based Split](#feature-engine-or-psi-based-split)
- [Detailed Preprocessing Pipeline Table](#detailed-preprocessing-pipeline-table)
- [Evaluation & Debugging](#evaluation--debugging)
- [Advanced Usage Example](#advanced-usage-example)
- [🛠️ Contributing](#-contributing)
- [📄 License](#-license)
- [🔧 Continuous Integration](#-continuous-integration)
- [📚 Resources](#-resources)

---

## 🚀 Installation

Ensure you have Python 3.6 or higher installed.

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/ghadfield32/ml_preprocessor.git
    cd ml_preprocessor
    ```

2. **Install the Package:**

    ```bash
    pip install .
    ```

    Alternatively, for development purposes:

    ```bash
    pip install -e .
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **(Optional) Install Custom Preprocessing Package Directly from GitHub:**

    ```bash
    pip install git+https://github.com/ghadfield32/ml_preprocessor.git
    ```

---

## 📝 Usage

### Basic Train Example

```python
from datapreprocessor import DataPreprocessor
import pandas as pd

# Load your dataset
df = pd.read_csv('path_to_your_dataset.csv')

# Initialize the DataPreprocessor in train mode
preprocessor = DataPreprocessor(
    model_type="Tree Based Classifier",
    y_variable=["target"],
    ordinal_categoricals=["ordinal_feature1", "ordinal_feature2"],
    nominal_categoricals=["nominal_feature1", "nominal_feature2"],
    numericals=["num_feature1", "num_feature2"],
    mode="train",
    # options=your_config_options,  # Uncomment if you wish to adjust settings via YAML config
    debug=True
)

# Execute preprocessing
X_train, X_test, y_train, y_test, recommendations, X_test_inverse = preprocessor.final_preprocessing(
    df.drop('target', axis=1), 
    df['target']
)
```

### Preprocessing Configuration

Adjust the preprocessing steps via the `preprocessor_config.yaml` file located in the `config/` directory. Customize parameters such as imputation strategies, encoding methods, scaling techniques, and SMOTE configurations based on your project needs.

### Running Preprocessing in Predict Mode

```python
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
    # options=your_config_options,  # Uncomment to adjust custom settings
    debug=True
)

# Execute preprocessing
X_preprocessed, recommendations, X_inversed = preprocessor.final_preprocessing(new_data)
```

### Running Preprocessing in Cluster Mode

```python
from datapreprocessor import DataPreprocessor
import pandas as pd

# Load your dataset for clustering
df = pd.read_csv('path_to_your_dataset.csv')

# Initialize the DataPreprocessor in cluster mode (e.g., for KMeans)
preprocessor = DataPreprocessor(
    model_type="Kmeans",
    y_variable=["target"],
    ordinal_categoricals=["ordinal_feature1", "ordinal_feature2"],
    nominal_categoricals=["nominal_feature1", "nominal_feature2"],
    numericals=["num_feature1", "num_feature2"],
    mode="cluster",
    # options=your_config_options,  # Uncomment for custom configurations
    debug=True
)

# Execute preprocessing
X_preprocessed, recommendations = preprocessor.final_preprocessing(df)
```

---

## ⏱ Time Series Modes

The package offers advanced modes for handling sequence-based (time series) data:

### set_window

```yaml
options:
  ts_sequence_mode: "set_window"
  sequence_modes:
    set_window:
      window_size: 10
      max_sequence_length: 10
```

- **Description:**  
  Defines a fixed window size for each sequence. Ideal for equally spaced time series.

### pad

```yaml
options:
  ts_sequence_mode: "pad"
  sequence_modes:
    pad:
      pad_threshold: 0.3
      padding_side: "post"
```

- **Description:**  
  Pads shorter sequences up to a maximum length. Adjusted by `pad_threshold` and `padding_side` (either `pre` or `post`).

### dtw

```yaml
options:
  ts_sequence_mode: "dtw"
  sequence_modes:
    dtw:
      reference_sequence: "max"
      dtw_threshold: 0.3
```

- **Description:**  
  Uses Dynamic Time Warping for sequence alignment. The `reference_sequence` can be specified (e.g., `max`, `mean`, or a custom identifier), and `dtw_threshold` controls match strictness.

---

## 🔀 Splitting Methods

Choose one of the following methods to split your dataset:

### Percentage-based Split

```yaml
split_dataset:
  test_size: 0.2
  random_state: 42
time_series_split:
  method: "standard"
```

- **Description:**  
  A common 80/20 train-test split with optional stratification.

### Date-based Split

```yaml
split_dataset:
  time_split_column: "datetime"
  time_split_value: "2025-02-14 11:50:00"
time_series_split:
  method: "standard"
```

- **Description:**  
  Splits data based on a specified datetime boundary.

### Sequence-aware Split

```yaml
time_series_split:
  method: "sequence_aware"
  split_date: "2025-02-14 11:50:00"  # Alternatively, use target_train_fraction
  target_train_fraction: 0.8
```

- **Description:**  
  Preserves entire sequences during split—ideal when sequences must remain intact.

### Feature-Engine or PSI-based Split

```yaml
psi_feature_selection:
  enabled: true
  threshold: 0.25
  split_frac: 0.75
  apply_before_split: true
feature_engine_split:
  enabled: true
  split_frac: 0.75
time_series_split:
  method: "feature_engine"
```

- **Description:**  
  Uses advanced feature-engineering based methods to determine the train/test split.

---

---

## 🚀 Advanced Usage Example (Time Series)

Below is an example that illustrates training an LSTM model using **pad** mode and a sequence-aware split for time series data.

```python
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datapreprocessor import DataPreprocessor

# Load time-series data
data = pd.read_parquet('time_series_data.parquet')

# Configure the DataPreprocessor for LSTM using pad mode with a sequence-aware split
preprocessor = DataPreprocessor(
    model_type="LSTM",
    y_variable=["target"],
    numericals=["feature1", "feature2"],
    mode="train",
    options={
        "time_column": "datetime",
        "use_horizon_sequence": True,
        "horizon_sequence_number": 1,
        "step_size": 1,
        "sequence_modes": {
            "pad": {
                "pad_threshold": 0.3,
                "padding_side": "post"
            }
        },
        "ts_sequence_mode": "pad",
        "time_series_split": {
            "method": "sequence_aware",
            "target_train_fraction": 0.8
        }
    },
    debug=True
)

# Analyze potential split points (if needed)
print("Analyzing potential split points...")
split_options = dtw_date_preprocessor.analyze_split_options(data)
for i, option in enumerate(split_options[:3]):  # Display top 3 options
   print(f"Option {i+1}: Split at {option['split_time']} - Train fraction: {option['train_fraction']:.2f}")

# Preprocess the time series data
X_train, X_test, y_train, y_test, _, _ = preprocessor.final_ts_preprocessing(data)

# Build and train the LSTM model
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)  # Assuming forecast horizon = 1
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=32)
```

---


## 📊 Detailed Preprocessing Pipeline Table

Below is a comprehensive table detailing the default preprocessing paths and options for various machine learning model types:

| **Preprocessing Step**          | **Tree Based Classifier**                                                                                                                                                                                                                                                         | **Logistic Regression**                                                                                                                                                                                                                                                                                        | **K-Means**                                                                                                                                                                                                                                                                                                         | **Linear Regression**                                                                                                                                                                                                                                                                                           | **Tree Based Regressor**                                                                                                                                                                                                                                                          | **Support Vector Machine**                                                                                                                                                                                                                                                                                                 | **Additional Options**                                                                                                                                                                                                                                                                                       |
|---------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **1. Data Splitting**           | - **Train/Test Split**<br>- **Stratified Split** (if `stratify_for_classification: true`)<br>- **Test Size:** 0.2<br>- **Random State:** 42                                                                                                                                                          | - **Train/Test Split**<br>- **Stratified Split** (if `stratify_for_classification: true`)<br>- **Test Size:** 0.2<br>- **Random State:** 42                                                                                                                                                                                              | - **No Splitting** (`split_dataset.test_size: null`)<br>- **Stratify:** false                                                                                                                                                                                                                                                                                        | - **Train/Test Split**<br>- **Stratified Split:** false<br>- **Test Size:** 0.2<br>- **Random State:** 42                                                                                                                                                                                                                                                                   | - **Train/Test Split**<br>- **Stratified Split:** false<br>- **Test Size:** 0.2<br>- **Random State:** 42                                                                                                                                                                                                                                                                  | - **Train/Test Split**<br>- **Stratified Split:** true<br>- **Test Size:** 0.2<br>- **Random State:** 42                                                                                                                                                                                                                                                                   | - **Custom Split Ratios**<br>- **Cross-Validation Setups**<br>- **Option to Disable Splitting** (e.g., for clustering)                                                                                                                                                                                                                                                                              |
| **2. Handling Missing Values**  | - **Numerical:** `SimpleImputer` (strategy: median)<br>- **Categorical:** `SimpleImputer` (strategy: most_frequent, fill_value: "Missing")                                                                                                                                             | - **Numerical:** `SimpleImputer` (strategy: mean)<br>- **Categorical:** `SimpleImputer` (strategy: most_frequent, fill_value: "Missing")                                                                                                                                                                                     | - **Numerical:** `SimpleImputer` (strategy: mean)<br>- **Categorical:** `SimpleImputer` (strategy: most_frequent, fill_value: "Missing")                                                                                                                                                                                                              | - **Numerical:** `SimpleImputer` (strategy: mean)<br>- **Categorical:** `SimpleImputer` (strategy: most_frequent, fill_value: "Missing")                                                                                                                                                                                                                 | - **Numerical:** `SimpleImputer` (strategy: median)<br>- **Categorical:** `SimpleImputer` (strategy: most_frequent, fill_value: "Missing")                                                                                                                                                                                                                 | - **Numerical:** `KNNImputer` (strategy: mean, k_neighbors: 5)<br>- **Categorical:** `SimpleImputer` (strategy: constant, fill_value: "Unknown")                                                                                                                                                                                                                                               | - **Advanced Imputation Methods:** e.g., Iterative Imputer<br>- **Custom Imputation Strategies**<br>- **Different Strategies per Feature or Feature Type**                                                                                                                                                                                                                   |
| **3. Testing for Normality**    | - **P-Value Threshold:** 0.05<br>- **Skewness Threshold:** 1.0<br>- **Tests Used:** Shapiro-Wilk (if sample size ≤ 5000) or Anderson-Darling (if sample size > 5000)<br>- **Transformation Needed:** if p-value < threshold **or** skewness > threshold                                                                                                                                                  | - **P-Value Threshold:** 0.05<br>- **Skewness Threshold:** 1.0<br>- **Tests Used:** Shapiro-Wilk or Anderson-Darling<br>- **Transformation Needed:** if p-value < threshold **or** skewness > threshold                                                                                                                                         | - **Not Applicable:** Typically skipped or handled differently for clustering                                                                                                                                                                                                                                                                                                    | - **P-Value Threshold:** 0.05<br>- **Skewness Threshold:** 1.0<br>- **Tests Used:** Shapiro-Wilk or Anderson-Darling<br>- **Transformation Needed:** if p-value < threshold **or** skewness > threshold                                                                                                                                                                                  | - **P-Value Threshold:** 0.05<br>- **Skewness Threshold:** 1.0<br>- **Tests Used:** Shapiro-Wilk or Anderson-Darling<br>- **Transformation Needed:** if p-value < threshold **or** skewness > threshold                                                                                                                                                                                  | - **P-Value Threshold:** 0.05<br>- **Skewness Threshold:** 1.0<br>- **Tests Used:** Shapiro-Wilk or Anderson-Darling<br>- **Transformation Needed:** if p-value < threshold **or** skewness > threshold                                                                                                                                                                                | - **Additional Normality Tests**<br>- **Different Thresholds**<br>- **Option to Skip Normality Testing**                                                                                                                                                                                                                                                                                             |
| **4. Handling Outliers**        | - **Outlier Detection:** IQR Filtering (`apply_iqr: true`)<br>- **Z-Score Filtering:** Disabled (`apply_zscore: false`)<br>- **Isolation Forest:** Disabled (`apply_isolation_forest: false`)<br>- **IQR Multiplier:** 1.5                                                  | - **Outlier Detection:** IQR Filtering (`apply_iqr: true`)<br>- **Z-Score Filtering:** Disabled (`apply_zscore: false`)<br>- **Isolation Forest:** Disabled (`apply_isolation_forest: false`)<br>- **IQR Multiplier:** 1.5                                                                         | - **Outlier Detection:** IQR Filtering (`apply_iqr: true`)<br>- **Z-Score Filtering:** Enabled (`apply_zscore: true`)<br>- **Isolation Forest:** Disabled (`apply_isolation_forest: false`)<br>- **IQR Multiplier:** 1.5<br>- **Z-Score Threshold:** 3                                                                                                                                                      | - **Outlier Detection:** IQR Filtering (`apply_iqr: true`)<br>- **Z-Score Filtering:** Enabled (`apply_zscore: true`)<br>- **Isolation Forest:** Disabled (`apply_isolation_forest: false`)<br>- **IQR Multiplier:** 1.5<br>- **Z-Score Threshold:** 3                                                                                                                                                     | - **Outlier Detection:** IQR Filtering (`apply_iqr: true`)<br>- **Z-Score Filtering:** Disabled (`apply_zscore: false`)<br>- **Isolation Forest:** Disabled (`apply_isolation_forest: false`)<br>- **IQR Multiplier:** 1.5                                                                                                                                            | - **Outlier Detection:** IQR Filtering (`apply_iqr: true`)<br>- **Z-Score Filtering:** Disabled (`apply_zscore: false`)<br>- **Isolation Forest:** Disabled (`apply_isolation_forest: false`)<br>- **IQR Multiplier:** 1.5                                                                                                                                            | - **Custom Threshold Settings**<br>- **Alternative Outlier Detection Methods:** e.g., Winsorization<br>- **Option to Apply Different Methods Based on Feature or Model Sensitivity**                                                                                                                                                                                                                   |
| **5. Encoding Categorical Features** | - **Ordinal Encoding:** `OrdinalEncoder`<br>- **Nominal Encoding:** `OneHotEncoder`<br>- **Frequency Encoding:** Not used<br>- **Handle Unknown:** ignore                                                                                                                         | - **Ordinal Encoding:** `OrdinalEncoder`<br>- **Nominal Encoding:** `OneHotEncoder`<br>- **Frequency Encoding:** Not used<br>- **Handle Unknown:** ignore                                                                                                                                                                       | - **Ordinal Encoding:** `OrdinalEncoder`<br>- **Nominal Encoding:** `OrdinalEncoder`<br>- **Frequency Encoding:** Not used<br>- **Handle Unknown:** ignore                                                                                                                                                                                                                | - **Ordinal Encoding:** `OrdinalEncoder`<br>- **Nominal Encoding:** `OneHotEncoder`<br>- **Frequency Encoding:** Not used<br>- **Handle Unknown:** ignore                                                                                                                                                                                                                         | - **Ordinal Encoding:** `OrdinalEncoder`<br>- **Nominal Encoding:** `OneHotEncoder`<br>- **Frequency Encoding:** Not used<br>- **Handle Unknown:** ignore                                                                                                                                                                                                                         | - **Ordinal Encoding:** `OrdinalEncoder`<br>- **Nominal Encoding:** `OneHotEncoder`<br>- **Frequency Encoding:** Not used<br>- **Handle Unknown:** ignore                                                                                                                                                                                                                             | - **Other Encoding Methods:** e.g., Binary Encoding, Hashing Encoding<br>- **Option to Disable Encoding for Certain Features**<br>- **Frequency Encoding for Nominal Features**                                                                                                                                                                                                                   |
| **6. Scaling Numerical Features** | - **Method:** `StandardScaler`<br>- **Features to Scale:** numericals<br>- **Default:** `StandardScaler` for classification<br>- **Options:** Can choose `StandardScaler`, `MinMaxScaler`, `RobustScaler`, or `None`                                                                         | - **Method:** `StandardScaler`<br>- **Features to Scale:** numericals<br>- **Default:** `StandardScaler` for classification<br>- **Options:** Can choose `StandardScaler`, `MinMaxScaler`, `RobustScaler`, or `None`                                                                                                                                  | - **Method:** `MinMaxScaler`<br>- **Features to Scale:** numericals<br>- **Default:** `MinMaxScaler` for clustering<br>- **Options:** Can choose `StandardScaler`, `MinMaxScaler`, `RobustScaler`, or `None`                                                                                                                                                                      | - **Method:** `StandardScaler`<br>- **Features to Scale:** numericals<br>- **Default:** `StandardScaler` for regression<br>- **Options:** Can choose `StandardScaler`, `MinMaxScaler`, `RobustScaler`, or `None`                                                                                                                                                                       | - **Method:** `None`<br>- **Features to Scale:** numericals (none scaled)<br>- **Default:** No scaling for tree-based regressors<br>- **Options:** Can choose `StandardScaler`, `MinMaxScaler`, `RobustScaler`, or `None`                                                                                                                                                             | - **Method:** `StandardScaler`<br>- **Features to Scale:** numericals<br>- **Default:** `StandardScaler` for classification<br>- **Options:** Can choose `StandardScaler`, `MinMaxScaler`, `RobustScaler`, or `None`                                                                                                                                                            | - **Custom Scaling Functions**<br>- **Power Transformer**<br>- **Option to Scale Only Specific Features**<br>- **Option to Disable Scaling**                                                                                                                                                                                                                                                                        |
| **7. Handling Class Imbalance** | - **SMOTE Variant:** `SMOTENC`<br>- **Applicable For:** Classification with mixed numerical and categorical features<br>- **Parameters:** `k_neighbors: 5`, `sampling_strategy: 'auto'`<br>- **Optional Variants:** `BorderlineSMOTE`, `ADASYN`, `SMOTEENN`, `SMOTETomek`<br>- **For Numerical Only Datasets:** `SMOTE` with optional variants | - **SMOTE Variant:** `SMOTENC`<br>- **Applicable For:** Classification with mixed numerical and categorical features<br>- **Parameters:** `k_neighbors: 5`, `sampling_strategy: 'auto'`<br>- **Optional Variants:** `BorderlineSMOTE`, `ADASYN`, `SMOTEENN`, `SMOTETomek`<br>- **For Numerical Only Datasets:** `SMOTE` with optional variants | - **Not Applicable:** Typically skipped or handled differently for clustering                                                                                                                                                                                                                                                                                                    | - **Not Applicable:** Not used for regression                                                                                                                                                                                                                                                                                           | - **Not Applicable:** Not used for regression                                                                                                                                                                                                                                                                                          | - **SMOTE Variant:** `SMOTENC`<br>- **Applicable For:** Classification with mixed numerical and categorical features<br>- **Parameters:** `k_neighbors: 5`, `sampling_strategy: 'auto'`<br>- **Optional Variants:** `BorderlineSMOTE`, `ADASYN`, `SMOTEENN`, `SMOTETomek`<br>- **For Numerical Only Datasets:** `SMOTE` with optional variants | - **Choice of SMOTE Variants Based on Dataset**<br>- **Option to Disable Class Imbalance Handling**<br>- **Customize `sampling_strategy` and `k_neighbors`**                                                                                                                                                                                                                                                                 |
| **8. Building Preprocessing Pipeline** | - **Pipeline Components:**<br>  - **Numerical:** `SimpleImputer` (median) → `StandardScaler`<br>  - **Ordinal:** `SimpleImputer` → `OrdinalEncoder`<br>  - **Nominal:** `SimpleImputer` → `OneHotEncoder`<br>- **Pipeline Type:** scikit-learn `Pipeline` & `ColumnTransformer`<br>- **Options:** Custom configurations, integration with other libraries | - **Pipeline Components:**<br>  - **Numerical:** `SimpleImputer` (mean) → `StandardScaler`<br>  - **Ordinal:** `SimpleImputer` → `OrdinalEncoder`<br>  - **Nominal:** `SimpleImputer` → `OneHotEncoder`<br>- **Pipeline Type:** scikit-learn `Pipeline` & `ColumnTransformer`<br>- **Options:** Custom configurations, integration with other libraries | - **Pipeline Components:**<br>  - **Numerical:** `SimpleImputer` (mean) → `MinMaxScaler`<br>  - **Ordinal:** `SimpleImputer` → `OrdinalEncoder`<br>  - **Nominal:** `SimpleImputer` → `OrdinalEncoder`<br>- **Pipeline Type:** scikit-learn `Pipeline` & `ColumnTransformer`<br>- **Options:** Custom configurations, integration with other libraries | - **Pipeline Components:**<br>  - **Numerical:** `SimpleImputer` (mean) → `StandardScaler`<br>  - **Ordinal:** `SimpleImputer` → `OrdinalEncoder`<br>  - **Nominal:** `SimpleImputer` → `OneHotEncoder`<br>- **Pipeline Type:** scikit-learn `Pipeline` & `ColumnTransformer`<br>- **Options:** Custom configurations, integration with other libraries | - **Pipeline Components:**<br>  - **Numerical:** `SimpleImputer` (median) → `None`<br>  - **Ordinal:** `SimpleImputer` → `OrdinalEncoder`<br>  - **Nominal:** `SimpleImputer` → `OneHotEncoder`<br>- **Pipeline Type:** scikit-learn `Pipeline` & `ColumnTransformer`<br>- **Options:** Custom configurations, integration with other libraries | - **Pipeline Components:**<br>  - **Numerical:** `KNNImputer` → `StandardScaler`<br>  - **Ordinal:** `SimpleImputer` → `OrdinalEncoder`<br>  - **Nominal:** `SimpleImputer` → `OneHotEncoder`<br>- **Pipeline Type:** scikit-learn `Pipeline` & `ColumnTransformer`<br>- **Options:** Custom configurations, integration with other libraries | - **Pipeline Flexibility:**<br>  - Utilize scikit-learn's `Pipeline` and `ColumnTransformer`<br>  - Integrate with libraries like TensorFlow or PyTorch<br>  - **Custom Configurations:** Adjust pipeline steps based on `model_type`<br>  - **Selective Steps:** Enable or disable certain steps as needed |
| **9. Saving & Loading Transformers** | - **Method:** Save using `joblib`<br>- **Path:** `transformers_dir/transformers.pkl`<br>- **Components Saved:** Numerical imputer, categorical imputer, preprocessor, SMOTE variant, feature order<br>- **Options:** Alternative serialization methods, versioning | - **Method:** Save using `joblib`<br>- **Path:** `transformers_dir/transformers.pkl`<br>- **Components Saved:** Numerical imputer, categorical imputer, preprocessor, SMOTE variant, feature order<br>- **Options:** Alternative serialization methods, versioning | - **Method:** Save using `joblib`<br>- **Path:** `transformers_dir/transformers.pkl`<br>- **Components Saved:** Numerical imputer, categorical imputer, preprocessor<br>- **Options:** Alternative serialization methods, versioning | - **Method:** Save using `joblib`<br>- **Path:** `transformers_dir/transformers.pkl`<br>- **Components Saved:** Numerical imputer, categorical imputer, preprocessor<br>- **Options:** Alternative serialization methods, versioning | - **Method:** Save using `joblib`<br>- **Path:** `transformers_dir/transformers.pkl`<br>- **Components Saved:** Numerical imputer, categorical imputer, preprocessor<br>- **Options:** Alternative serialization methods, versioning | - **Method:** Save using `joblib`<br>- **Path:** `transformers_dir/transformers.pkl`<br>- **Components Saved:** Numerical imputer, categorical imputer, preprocessor<br>- **Options:** Alternative serialization methods, versioning | - **Alternative Methods:** e.g., using `pickle`<br>- **Versioning:** Track transformer versions<br>- **Custom Save Paths:** Specify directories/filenames |
| **10. Inverse Transformations**   | - **Method:** Inverse scaling (`StandardScaler`)<br>- **Inverse Encoding:** `OrdinalEncoder` & `OneHotEncoder`<br>- **Purpose:** Reconstruct original feature values<br>- **Options:** Custom inverse logic, visualization of original vs. transformed data | - **Method:** Inverse scaling (`StandardScaler`)<br>- **Inverse Encoding:** `OrdinalEncoder` & `OneHotEncoder`<br>- **Purpose:** Reconstruct original feature values<br>- **Options:** Custom inverse logic, visualization of original vs. transformed data | - **Method:** Inverse scaling (`MinMaxScaler`)<br>- **Inverse Encoding:** `OrdinalEncoder`<br>- **Purpose:** Reconstruct original feature values<br>- **Options:** Custom inverse logic, visualization of original vs. transformed data | - **Method:** Inverse scaling (`StandardScaler`)<br>- **Inverse Encoding:** `OrdinalEncoder` & `OneHotEncoder`<br>- **Purpose:** Reconstruct original feature values<br>- **Options:** Custom inverse logic, visualization of original vs. transformed data | - **Method:** Inverse scaling: `None`<br>- **Inverse Encoding:** `OrdinalEncoder` & `OneHotEncoder`<br>- **Purpose:** Reconstruct original feature values<br>- **Options:** Custom inverse logic, visualization of original vs. transformed data | - **Method:** Inverse scaling (`StandardScaler`)<br>- **Inverse Encoding:** `OrdinalEncoder` & `OneHotEncoder`<br>- **Purpose:** Reconstruct original feature values<br>- **Options:** Custom inverse logic, visualization of original vs. transformed data | - **Customized Logic:** Specific rules for inversion<br>- **Visualization:** Compare original vs. transformed data<br>- **Selective Inversion:** Optionally invert only selected features |
| **11. Configuration Management** | - **Method:** YAML configuration files<br>- **Dynamic Adjustment:** Based on `model_type`<br>- **Options:** Environment-specific configs, parameter tuning through files                                                              | - **Method:** YAML configuration files<br>- **Dynamic Adjustment:** Based on `model_type`<br>- **Options:** Environment-specific configs, parameter tuning through files                                                              | - **Method:** YAML configuration files<br>- **Dynamic Adjustment:** Based on `model_type`<br>- **Options:** Environment-specific configs, parameter tuning through files                                                              | - **Method:** YAML configuration files<br>- **Dynamic Adjustment:** Based on `model_type`<br>- **Options:** Environment-specific configs, parameter tuning through files                                                              | - **Method:** YAML configuration files<br>- **Dynamic Adjustment:** Based on `model_type`<br>- **Options:** Environment-specific configs, parameter tuning through files                                                              | - **Method:** YAML configuration files<br>- **Dynamic Adjustment:** Based on `model_type`<br>- **Options:** Environment-specific configs, parameter tuning through files                                                              | - **Environment-Specific Configurations:** Different settings for development, production, etc.<br>- **Parameter Tuning:** Easily adjust via YAML<br>- **Easy Adjustments:** Update YAML to modify preprocessing steps per model type |

🔍 **Explanation of Columns**  
- **Preprocessing Step:** Sequential step in the workflow.  
- **Model Types:** Each column details the preprocessing techniques for a specific model type.  
- **Additional Options:** Other methods and configurations available for customization.

🛠️ **How to Use This Table**  
- **Identify the Steps:** Review each step to understand the actions required.  
- **Select Model-Specific Techniques:** Choose techniques recommended for your model (e.g., Logistic Regression).  
- **Explore Additional Options:** Consider alternate methods to further customize your workflow.

🔧 **Customization Tips**  
- **YAML Configuration:** Use the `preprocessor_config.yaml` file to seamlessly adjust parameters without changing code.  
- **Pipeline Flexibility:** Utilize scikit-learn’s `Pipeline` and `ColumnTransformer` for modular workflows.  
- **Imputation Strategies:** Experiment with different imputation methods (e.g., KNNImputer) based on model sensitivity.  
- **Encoding Choices:** Choose appropriate encoding (OneHotEncoder for nominal features in Logistic Regression, etc.).  
- **Scaling Methods:** Select scalers (e.g., MinMaxScaler for K-Means) to ensure features contribute equally.  
- **Handling Class Imbalance:** Use suitable SMOTE variants based on your dataset’s feature composition.  
- **Inverse Transformations:** Leverage these to interpret predictions by mapping scaled/encoded values back to original forms.

---

## 🔍 Evaluation & Debugging

The DataPreprocessor offers built-in debugging functionalities. Set the `debug` flag to `True` during initialization to print detailed logs and track the data flow through each step. This feature aids in troubleshooting and verifying preprocessing steps.



## 🛠️ Contributing

Contributions are welcome! Please fork the repository and submit a pull request with any enhancements or bug fixes.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🔧 Continuous Integration

The project utilizes continuous integration pipelines to run tests and ensure code quality. Check the repository settings for more details.

---

## 📚 Resources

- **Creating Great README Files for Your Python Projects:**  
  Learn best practices for crafting effective README files.
- **Awesome Badges:**  
  A repository of useful badges to enhance your README.
- **GitHub README Templates:**  
  Explore various README templates for inspiration.

---

*This README combines detailed information on both general machine learning data preprocessing and specific time series processing capabilities to provide a comprehensive guide for DataPreprocessor users.*
```

---

This final combined README includes all the original sections and examples—basic usage, advanced time series processing, and a detailed table outlining model-specific preprocessing steps—ensuring that users have a complete reference for utilizing the DataPreprocessor package.