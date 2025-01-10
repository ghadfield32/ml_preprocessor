# DataPreprocessor

**DataPreprocessor** is a comprehensive Python package designed to streamline the data preprocessing workflow for various machine learning models. It offers functionalities such as handling missing values, outlier detection, feature encoding, scaling, and addressing class imbalances using techniques like SMOTE. 

**Datapreprocessor is for after feature engineering/multicollinearity checks/data retrieval, ensure you've done this if appropriate for the model type you are working with**

## 🧰 Features

- **Flexible Preprocessing Pipeline:** Customize preprocessing steps based on your model type and mode, default path for most Regressive/Classifier/Clustering models that can be adjusted
- **Automatic Outlier Detection:** Utilize methods like Z-Score Filtering, IQR Filtering, and Isolation Forest.
- **Categorical Encoding:** Support for Ordinal and One-Hot Encoding.
- **Scaling Options:** Apply StandardScaler, MinMaxScaler, or RobustScaler.
- **Class Imbalance Handling:** Automated SMOTE/SMOTENC/SMOTEN handling. We have numerical + categorical datasets have SMOTENC applied, Categorical only has SMOTEN, and Numerical only goes through a criteria to use SMOTE/BorderlineSmote/ADSYN/SMOTEENN/SMOTETomek (these are optional also if you wanted to try a different one if you have a Numerical ONLY dataset)
- **Configuration Driven:** Easily adjust preprocessing steps via YAML configuration files.
- **Inverse Transformation:** Reconstruct original feature values from transformed data for interpretability.

## 📋 Table of Contents

- [🧰 Features](#-features)
- [🚀 Installation](#-installation)
- [📝 Usage](#-usage)
  - [Basic Example](#basic-example)
  - [Configuration](#configuration)
  - [Running Preprocessing in Predict Mode](#running-preprocessing-in-predict-mode)
- [🧪 Testing](#-testing)
- [🛠️ Contributing](#️-contributing)
- [📄 License](#-license)
- [🔧 Continuous Integration](#-continuous-integration)

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

4. **Install Custom Preprocessing Package:**

    To include the custom preprocessing package directly from GitHub, use:

    ```bash
    pip install git+https://github.com/ghadfield32/ml_preprocessor.git
    ```

## 📝 Usage

### Basic Train Example

```python
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
    # options=your_config_options,  # uncomment if custom scaling/encoding/transforming or want to adjust paths or anything
    debug=True
)

# Execute Preprocessing
X_train, X_test, y_train, y_test, recommendations, X_test_inverse = preprocessor.final_preprocessing(
    df.drop('target', axis=1), 
    df['target']
)
```

### Train Preprocessing Adjustment Configuration

Customize preprocessing steps via the `preprocessor_config.yaml` file located in the `config/` directory. Adjust parameters like imputation strategies, encoding methods, scaling techniques, and SMOTE configurations based on your project needs.

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
    # options=your_config_options,  # uncomment if custom scaling/encoding/transforming or want to adjust paths or anything
    debug=True
)

# Execute Preprocessing
X_preprocessed, recommendations, X_inversed = preprocessor.final_preprocessing(new_data)
```

### Predict Preprocessing Adjustment Configuration

Customize preprocessing steps via the `preprocessor_config.yaml`. Ensure that predict paths are same as training and if custom Scaling/Encoding/Transforming, then just point to the folder where those are located


### Running Preprocessing in Cluster Mode

```python
from datapreprocessor import DataPreprocessor
import pandas as pd

# Load new data for prediction
new_data = pd.read_csv('path_to_new_data.csv')

# Initialize the DataPreprocessor in predict mode
preprocessor = DataPreprocessor(
    model_type="Kmeans",
    y_variable=["target"],
    ordinal_categoricals=["ordinal_feature1", "ordinal_feature2"],
    nominal_categoricals=["nominal_feature1", "nominal_feature2"],
    numericals=["num_feature1", "num_feature2"],
    mode="cluster",
    # options=your_config_options,  # uncomment if custom scaling/encoding/transforming or want to adjust paths or anything
    debug=True
)

# Execute Preprocessing
X_preprocessed, recommendations = preprocessor.final_preprocessing(new_data)
```
## 🧪 Default Path for Each Model Type and Options:
| **Preprocessing Step**          | **Tree Based Classifier**                                                                                                                                                                                                                                                         | **Logistic Regression**                                                                                                                                                                                                                                                                                        | **K-Means**                                                                                                                                                                                                                                                                                                         | **Linear Regression**                                                                                                                                                                                                                                                                                           | **Tree Based Regressor**                                                                                                                                                                                                                                                          | **Support Vector Machine**                                                                                                                                                                                                                                                                                                 | **Additional Options**                                                                                                                                                                                                                                                                                       |
|---------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **1. Data Splitting**           | - **Train/Test Split**<br>- **Stratified Split** (if `stratify_for_classification: true`)<br>- **Test Size:** 0.2<br>- **Random State:** 42                                                                                                                                                          | - **Train/Test Split**<br>- **Stratified Split** (if `stratify_for_classification: true`)<br>- **Test Size:** 0.2<br>- **Random State:** 42                                                                                                                                                                                              | - **No Splitting** (`split_dataset.test_size: null`)<br>- **Stratify:** false                                                                                                                                                                                                                                                                                        | - **Train/Test Split**<br>- **Stratified Split:** false<br>- **Test Size:** 0.2<br>- **Random State:** 42                                                                                                                                                                                                                                                                   | - **Train/Test Split**<br>- **Stratified Split:** false<br>- **Test Size:** 0.2<br>- **Random State:** 42                                                                                                                                                                                                                                                                  | - **Train/Test Split**<br>- **Stratified Split:** true<br>- **Test Size:** 0.2<br>- **Random State:** 42                                                                                                                                                                                                                                                                   | - **Custom Split Ratios**<br>- **Cross-Validation Setups**<br>- **Option to Disable Splitting** (e.g., for clustering)                                                                                                                                                                                                                                                                              |
| **2. Handling Missing Values**  | - **Numerical:** `SimpleImputer` (strategy: median)<br>- **Categorical:** `SimpleImputer` (strategy: most_frequent, fill_value: "Missing")                                                                                                                                             | - **Numerical:** `SimpleImputer` (strategy: mean)<br>- **Categorical:** `SimpleImputer` (strategy: most_frequent, fill_value: "Missing")                                                                                                                                                                                     | - **Numerical:** `SimpleImputer` (strategy: mean)<br>- **Categorical:** `SimpleImputer` (strategy: most_frequent, fill_value: "Missing")                                                                                                                                                                                                              | - **Numerical:** `SimpleImputer` (strategy: mean)<br>- **Categorical:** `SimpleImputer` (strategy: most_frequent, fill_value: "Missing")                                                                                                                                                                                                                 | - **Numerical:** `SimpleImputer` (strategy: median)<br>- **Categorical:** `SimpleImputer` (strategy: most_frequent, fill_value: "Missing")                                                                                                                                                                                                                 | - **Numerical:** `KNNImputer` (strategy: mean, imputer: KNNImputer, k_neighbors: 5)<br>- **Categorical:** `SimpleImputer` (strategy: constant, fill_value: "Unknown")                                                                                                                                                                               | - **Advanced Imputation Methods:** e.g., Iterative Imputer<br>- **Custom Imputation Strategies**<br>- **Different Strategies per Feature or Feature Type**                                                                                                                                                                                                                   |
| **3. Testing for Normality**    | - **P-Value Threshold:** 0.05<br>- **Skewness Threshold:** 1.0<br>- **Tests Used:** Shapiro-Wilk (if sample size ≤ 5000) or Anderson-Darling (if sample size > 5000)<br>- **Transformation Needed:** if p-value < threshold **or** skewness > threshold                                                                                                                                                  | - **P-Value Threshold:** 0.05<br>- **Skewness Threshold:** 1.0<br>- **Tests Used:** Shapiro-Wilk or Anderson-Darling<br>- **Transformation Needed:** if p-value < threshold **or** skewness > threshold                                                                                                                                         | - **Not Applicable:** Typically skipped or handled differently for clustering                                                                                                                                                                                                                                                                                                    | - **P-Value Threshold:** 0.05<br>- **Skewness Threshold:** 1.0<br>- **Tests Used:** Shapiro-Wilk or Anderson-Darling<br>- **Transformation Needed:** if p-value < threshold **or** skewness > threshold                                                                                                                                                                                  | - **P-Value Threshold:** 0.05<br>- **Skewness Threshold:** 1.0<br>- **Tests Used:** Shapiro-Wilk or Anderson-Darling<br>- **Transformation Needed:** if p-value < threshold **or** skewness > threshold                                                                                                                                                                                  | - **P-Value Threshold:** 0.05<br>- **Skewness Threshold:** 1.0<br>- **Tests Used:** Shapiro-Wilk or Anderson-Darling<br>- **Transformation Needed:** if p-value < threshold **or** skewness > threshold                                                                                                                                                                                | - **Additional Normality Tests**<br>- **Different Thresholds**<br>- **Option to Skip Normality Testing**                                                                                                                                                                                                                                                                                             |
| **4. Handling Outliers**        | - **Outlier Detection:** IQR Filtering (`apply_iqr: true`)<br>- **Z-Score Filtering:** Disabled (`apply_zscore: false`)<br>- **Isolation Forest:** Disabled (`apply_isolation_forest: false`)<br>- **IQR Multiplier:** 1.5                                                  | - **Outlier Detection:** IQR Filtering (`apply_iqr: true`)<br>- **Z-Score Filtering:** Disabled (`apply_zscore: false`)<br>- **Isolation Forest:** Disabled (`apply_isolation_forest: false`)<br>- **IQR Multiplier:** 1.5                                                                         | - **Outlier Detection:** IQR Filtering (`apply_iqr: true`)<br>- **Z-Score Filtering:** Enabled (`apply_zscore: true`)<br>- **Isolation Forest:** Disabled (`apply_isolation_forest: false`)<br>- **IQR Multiplier:** 1.5<br>- **Z-Score Threshold:** 3                                                                                                                                                      | - **Outlier Detection:** IQR Filtering (`apply_iqr: true`)<br>- **Z-Score Filtering:** Enabled (`apply_zscore: true`)<br>- **Isolation Forest:** Disabled (`apply_isolation_forest: false`)<br>- **IQR Multiplier:** 1.5<br>- **Z-Score Threshold:** 3                                                                                                                                                     | - **Outlier Detection:** IQR Filtering (`apply_iqr: true`)<br>- **Z-Score Filtering:** Disabled (`apply_zscore: false`)<br>- **Isolation Forest:** Disabled (`apply_isolation_forest: false`)<br>- **IQR Multiplier:** 1.5                                                                                                                                            | - **Outlier Detection:** IQR Filtering (`apply_iqr: true`)<br>- **Z-Score Filtering:** Disabled (`apply_zscore: false`)<br>- **Isolation Forest:** Disabled (`apply_isolation_forest: false`)<br>- **IQR Multiplier:** 1.5                                                                                                                                            | - **Custom Threshold Settings**<br>- **Alternative Outlier Detection Methods:** e.g., Winsorization<br>- **Option to Apply Different Methods Based on Feature or Model Sensitivity**                                                                                                                                                                                                                   |
| **5. Encoding Categorical Features** | - **Ordinal Encoding:** `OrdinalEncoder`<br>- **Nominal Encoding:** `OneHotEncoder`<br>- **Frequency Encoding:** Not used<br>- **Handle Unknown:** ignore                                                                                                                         | - **Ordinal Encoding:** `OrdinalEncoder`<br>- **Nominal Encoding:** `OneHotEncoder`<br>- **Frequency Encoding:** Not used<br>- **Handle Unknown:** ignore                                                                                                                                                                       | - **Ordinal Encoding:** `OrdinalEncoder`<br>- **Nominal Encoding:** `OrdinalEncoder`<br>- **Frequency Encoding:** Not used<br>- **Handle Unknown:** ignore                                                                                                                                                                                                                | - **Ordinal Encoding:** `OrdinalEncoder`<br>- **Nominal Encoding:** `OneHotEncoder`<br>- **Frequency Encoding:** Not used<br>- **Handle Unknown:** ignore                                                                                                                                                                                                                         | - **Ordinal Encoding:** `OrdinalEncoder`<br>- **Nominal Encoding:** `OneHotEncoder`<br>- **Frequency Encoding:** Not used<br>- **Handle Unknown:** ignore                                                                                                                                                                                                                         | - **Ordinal Encoding:** `OrdinalEncoder`<br>- **Nominal Encoding:** `OneHotEncoder`<br>- **Frequency Encoding:** Not used<br>- **Handle Unknown:** ignore                                                                                                                                                                                                                             | - **Other Encoding Methods:** e.g., Binary Encoding, Hashing Encoding<br>- **Option to Disable Encoding for Certain Features**<br>- **Frequency Encoding for Nominal Features**                                                                                                                                                                                                                   |
| **6. Scaling Numerical Features** | - **Method:** `StandardScaler`<br>- **Features to Scale:** numericals<br>- **Default:** `StandardScaler` for classification<br>- **Options:** Can choose `StandardScaler`, `MinMaxScaler`, `RobustScaler`, or `None`                                                                         | - **Method:** `StandardScaler`<br>- **Features to Scale:** numericals<br>- **Default:** `StandardScaler` for classification<br>- **Options:** Can choose `StandardScaler`, `MinMaxScaler`, `RobustScaler`, or `None`                                                                                                                                  | - **Method:** `MinMaxScaler`<br>- **Features to Scale:** numericals<br>- **Default:** `MinMaxScaler` for clustering<br>- **Options:** Can choose `StandardScaler`, `MinMaxScaler`, `RobustScaler`, or `None`                                                                                                                                                                      | - **Method:** `StandardScaler`<br>- **Features to Scale:** numericals<br>- **Default:** `StandardScaler` for regression<br>- **Options:** Can choose `StandardScaler`, `MinMaxScaler`, `RobustScaler`, or `None`                                                                                                                                                                       | - **Method:** `None`<br>- **Features to Scale:** numericals (none scaled)<br>- **Default:** No scaling for tree-based regressors<br>- **Options:** Can choose `StandardScaler`, `MinMaxScaler`, `RobustScaler`, or `None`                                                                                                                                                             | - **Method:** `StandardScaler`<br>- **Features to Scale:** numericals<br>- **Default:** `StandardScaler` for classification<br>- **Options:** Can choose `StandardScaler`, `MinMaxScaler`, `RobustScaler`, or `None`                                                                                                                                                            | - **Custom Scaling Functions**<br>- **Power Transformer**<br>- **Option to Scale Only Specific Features**<br>- **Option to Disable Scaling**                                                                                                                                                                                                                                                                        |
| **7. Handling Class Imbalance** | - **SMOTE Variant:** `SMOTENC`<br>- **Applicable For:** Classification with mixed numerical and categorical features<br>- **Parameters:** `k_neighbors: 5`, `sampling_strategy: 'auto'`<br>- **Optional Variants:** `BorderlineSMOTE`, `ADASYN`, `SMOTEENN`, `SMOTETomek`<br>- **For Numerical Only Datasets:** `SMOTE` with optional variants | - **SMOTE Variant:** `SMOTENC`<br>- **Applicable For:** Classification with mixed numerical and categorical features<br>- **Parameters:** `k_neighbors: 5`, `sampling_strategy: 'auto'`<br>- **Optional Variants:** `BorderlineSMOTE`, `ADASYN`, `SMOTEENN`, `SMOTETomek`<br>- **For Numerical Only Datasets:** `SMOTE` with optional variants | - **Not Applicable:** Typically skipped or handled differently for clustering                                                                                                                                                                                                                                                                                                    | - **Not Applicable:** Not used for regression                                                                                                                                                                                                                                                                                           | - **Not Applicable:** Not used for regression                                                                                                                                                                                                                                                                                          | - **SMOTE Variant:** `SMOTENC`<br>- **Applicable For:** Classification with mixed numerical and categorical features<br>- **Parameters:** `k_neighbors: 5`, `sampling_strategy: 'auto'`<br>- **Optional Variants:** `BorderlineSMOTE`, `ADASYN`, `SMOTEENN`, `SMOTETomek`<br>- **For Numerical Only Datasets:** `SMOTE` with optional variants | - **Choice of SMOTE Variants Based on Dataset**<br>- **Option to Disable Class Imbalance Handling**<br>- **Customize `sampling_strategy` and `k_neighbors`**                                                                                                                                                                                                                                                                 |
| **8. Building Preprocessing Pipeline** | - **Pipeline Components:**<br>  - **Numerical:** `SimpleImputer` (median) → `StandardScaler`<br>  - **Ordinal:** `SimpleImputer` → `OrdinalEncoder`<br>  - **Nominal:** `SimpleImputer` → `OneHotEncoder`<br>- **Pipeline Type:** scikit-learn `Pipeline` & `ColumnTransformer`<br>- **Options:** Custom pipeline configurations, Integration with other libraries | - **Pipeline Components:**<br>  - **Numerical:** `SimpleImputer` (mean) → `StandardScaler`<br>  - **Ordinal:** `SimpleImputer` → `OrdinalEncoder`<br>  - **Nominal:** `SimpleImputer` → `OneHotEncoder`<br>- **Pipeline Type:** scikit-learn `Pipeline` & `ColumnTransformer`<br>- **Options:** Custom pipeline configurations, Integration with other libraries | - **Pipeline Components:**<br>  - **Numerical:** `SimpleImputer` (mean) → `MinMaxScaler`<br>  - **Ordinal:** `SimpleImputer` → `OrdinalEncoder`<br>  - **Nominal:** `SimpleImputer` → `OrdinalEncoder`<br>- **Pipeline Type:** scikit-learn `Pipeline` & `ColumnTransformer`<br>- **Options:** Custom pipeline configurations, Integration with other libraries | - **Pipeline Components:**<br>  - **Numerical:** `SimpleImputer` (mean) → `StandardScaler`<br>  - **Ordinal:** `SimpleImputer` → `OrdinalEncoder`<br>  - **Nominal:** `SimpleImputer` → `OneHotEncoder`<br>- **Pipeline Type:** scikit-learn `Pipeline` & `ColumnTransformer`<br>- **Options:** Custom pipeline configurations, Integration with other libraries | - **Pipeline Components:**<br>  - **Numerical:** `SimpleImputer` (median) → `None`<br>  - **Ordinal:** `SimpleImputer` → `OrdinalEncoder`<br>  - **Nominal:** `SimpleImputer` → `OneHotEncoder`<br>- **Pipeline Type:** scikit-learn `Pipeline` & `ColumnTransformer`<br>- **Options:** Custom pipeline configurations, Integration with other libraries | - **Pipeline Components:**<br>  - **Numerical:** `KNNImputer` → `StandardScaler`<br>  - **Ordinal:** `SimpleImputer` → `OrdinalEncoder`<br>  - **Nominal:** `SimpleImputer` → `OneHotEncoder`<br>- **Pipeline Type:** scikit-learn `Pipeline` & `ColumnTransformer`<br>- **Options:** Custom pipeline configurations, Integration with other libraries | - **Pipeline Flexibility:**<br>  - Use scikit-learn's `Pipeline` and `ColumnTransformer`<br>  - Integrate with other libraries like TensorFlow, PyTorch<br>  - **Custom Configurations:** Adjust pipeline steps based on `model_type`<br>  - **Option to Include/Exclude Steps:** Enable or disable certain preprocessing steps as needed |
| **9. Saving & Loading Transformers** | - **Method:** Save using `joblib`<br>- **Path:** `transformers_dir/transformers.pkl`<br>- **Components Saved:** Numerical imputer, categorical imputer, preprocessor, SMOTE variant, final feature order<br>- **Options:** Alternative serialization methods, Versioning Transformers      | - **Method:** Save using `joblib`<br>- **Path:** `transformers_dir/transformers.pkl`<br>- **Components Saved:** Numerical imputer, categorical imputer, preprocessor, SMOTE variant, final feature order<br>- **Options:** Alternative serialization methods, Versioning Transformers      | - **Method:** Save using `joblib`<br>- **Path:** `transformers_dir/transformers.pkl`<br>- **Components Saved:** Numerical imputer, categorical imputer, preprocessor<br>- **Options:** Alternative serialization methods, Versioning Transformers                                                                             | - **Method:** Save using `joblib`<br>- **Path:** `transformers_dir/transformers.pkl`<br>- **Components Saved:** Numerical imputer, categorical imputer, preprocessor<br>- **Options:** Alternative serialization methods, Versioning Transformers                                                                             | - **Method:** Save using `joblib`<br>- **Path:** `transformers_dir/transformers.pkl`<br>- **Components Saved:** Numerical imputer, categorical imputer, preprocessor<br>- **Options:** Alternative serialization methods, Versioning Transformers                                                                             | - **Method:** Save using `joblib`<br>- **Path:** `transformers_dir/transformers.pkl`<br>- **Components Saved:** Numerical imputer, categorical imputer, preprocessor<br>- **Options:** Alternative serialization methods, Versioning Transformers                                                                             | - **Alternative Serialization Methods:** e.g., `pickle`<br>- **Versioning Transformers:** Track different versions of transformers<br>- **Custom Save Paths:** Specify different directories or filenames for transformers                                                                                                                                   |
| **10. Inverse Transformations**   | - **Method:** Inverse scaling (`StandardScaler`)<br>- **Inverse Encoding:** `OrdinalEncoder` and `OneHotEncoder`<br>- **Purpose:** Reconstruct original feature values for interpretability<br>- **Options:** Customized inverse logic, Visualization of original vs. transformed data | - **Method:** Inverse scaling (`StandardScaler`)<br>- **Inverse Encoding:** `OrdinalEncoder` and `OneHotEncoder`<br>- **Purpose:** Reconstruct original feature values for interpretability<br>- **Options:** Customized inverse logic, Visualization of original vs. transformed data | - **Method:** Inverse scaling (`MinMaxScaler`)<br>- **Inverse Encoding:** `OrdinalEncoder`<br>- **Purpose:** Reconstruct original feature values for interpretability<br>- **Options:** Customized inverse logic, Visualization of original vs. transformed data | - **Method:** Inverse scaling (`StandardScaler`)<br>- **Inverse Encoding:** `OrdinalEncoder` and `OneHotEncoder`<br>- **Purpose:** Reconstruct original feature values for interpretability<br>- **Options:** Customized inverse logic, Visualization of original vs. transformed data | - **Method:** Inverse scaling: `None`<br>- **Inverse Encoding:** `OrdinalEncoder` and `OneHotEncoder`<br>- **Purpose:** Reconstruct original feature values for interpretability<br>- **Options:** Customized inverse logic, Visualization of original vs. transformed data | - **Method:** Inverse scaling (`StandardScaler`)<br>- **Inverse Encoding:** `OrdinalEncoder` and `OneHotEncoder`<br>- **Purpose:** Reconstruct original feature values for interpretability<br>- **Options:** Customized inverse logic, Visualization of original vs. transformed data | - **Customized Inverse Logic:** Specific rules or methods for inverse transformations<br>- **Visualization:** Compare original and transformed data<br>- **Selective Inversion:** Choose which features to invert |
| **11. Configuration Management** | - **Method:** YAML configuration files<br>- **Dynamic Adjustment:** Based on `model_type`<br>- **Options:** Environment-specific configurations, Parameter tuning through configuration files                                                              | - **Method:** YAML configuration files<br>- **Dynamic Adjustment:** Based on `model_type`<br>- **Options:** Environment-specific configurations, Parameter tuning through configuration files                                                              | - **Method:** YAML configuration files<br>- **Dynamic Adjustment:** Based on `model_type`<br>- **Options:** Environment-specific configurations, Parameter tuning through configuration files                                                              | - **Method:** YAML configuration files<br>- **Dynamic Adjustment:** Based on `model_type`<br>- **Options:** Environment-specific configurations, Parameter tuning through configuration files                                                              | - **Method:** YAML configuration files<br>- **Dynamic Adjustment:** Based on `model_type`<br>- **Options:** Environment-specific configurations, Parameter tuning through configuration files                                                              | - **Method:** YAML configuration files<br>- **Dynamic Adjustment:** Based on `model_type`<br>- **Options:** Environment-specific configurations, Parameter tuning through configuration files                                                              | - **Environment-Specific Configurations:** Different settings for different environments (e.g., development, production)<br>- **Parameter Tuning:** Adjust preprocessing parameters via config files<br>- **Easy Adjustments:** Update YAML to change preprocessing steps per `model_type`                                                                                                                                                                                                |


🔍 Explanation of Columns

    Preprocessing Step: The sequential step in the data preprocessing workflow.

    Model Types: Each column corresponds to a specific machine learning model type, detailing the preprocessing techniques applied at each step.

    Additional Options: Other available methods and configurations that can be customized or extended based on specific project needs.

🛠️ How to Use This Table

    Identify the Steps: Review each preprocessing step to understand the necessary actions in the workflow.

    Select Model-Specific Techniques: For your specific model type (e.g., Logistic Regression), identify the recommended techniques.

    Explore Additional Options: Consider additional methods and configurations to further customize the preprocessing pipeline according to your dataset and project requirements.

🔧 Customization Tips

    YAML Configuration: Leverage the preprocessor_config.yaml file to adjust preprocessing parameters seamlessly. This allows you to tailor the preprocessing steps to different model types without modifying the code.

    Pipeline Flexibility: Utilize scikit-learn's Pipeline and ColumnTransformer to create reusable and modular preprocessing workflows. This enhances maintainability and scalability.

    Imputation Strategies: Experiment with different imputation methods to handle missing data effectively based on data distribution and model sensitivity. For example, using KNNImputer for models sensitive to feature scaling like Support Vector Machine.

    Encoding Choices: Choose encoding techniques that best capture the relationships in your categorical data without introducing multicollinearity. For instance, using OneHotEncoder for nominal features in Logistic Regression.

    Scaling Methods: Select scaling methods that align with your model's requirements. For example, MinMaxScaler is preferred for K-Means Clustering to ensure all features contribute equally to distance calculations.

    Handling Class Imbalance: Select appropriate SMOTE variants based on your dataset's feature composition. Use SMOTENC for datasets with both numerical and categorical features in classification tasks, SMOTEN for purely categorical datasets, and SMOTE for purely numerical datasets. Additionally, you can choose other variants like BorderlineSMOTE, ADASYN, SMOTEENN, or SMOTETomek if further customization is needed.

    Inverse Transformations: Utilize inverse transformations for interpretability of model predictions. This is especially useful in understanding how scaled or encoded features relate back to the original data.
    

## 🛠️ Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## 📄 License

This project is licensed under the MIT License.

🛠️ Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.
📄 License

This project is licensed under the MIT License.
📚 Resources

    Creating Great README Files for Your Python Projects:
    Learn best practices for crafting effective README files.
    Read the Tutorial

    Awesome Badges:
    A repository of useful badges for your README.
    Check it out

    GitHub README Templates:
    Explore various README templates to inspire your own.
    Visit GitHub README Templates