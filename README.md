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

Ensure you have Python 3.6 or higher installed.

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
Basic Example:

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
X_train, X_test, y_train, y_test, recommendations, X_test_inverse = preprocessor.preprocess_train(df.drop('target', axis=1), df['target'])

Configuration:

Customize preprocessing steps via the preprocessor_config.yaml file located in the config/ directory. Adjust parameters like imputation strategies, encoding methods, scaling techniques, and SMOTE configurations based on your project needs.
Running Preprocessing in Predict Mode:

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

🧪 Testing

Run unit tests using pytest to ensure the package functions as expected:

pytest

🛠️ Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.
📄 License

This project is licensed under the MIT License.


### **Explanation:**

- **Sections:** Clearly divided into Features, Installation, Usage, Testing, Contributing, and License.
- **Code Blocks:** Provide clear examples of how to use the package.
- **Customization:** Encourages users to adjust configurations as needed.

---

## **Step 10: Implement Continuous Integration (CI) with GitHub Actions**

### **Purpose:**
Automate testing to ensure that your package remains functional with every change. This helps in maintaining code quality and catching issues early.

### **Action Items:**
1. **Define CI Workflow:**
   - Use GitHub Actions to automate testing on every push and pull request.
2. **Ensure Environment Consistency:**
   - Use specified Python versions.
3. **Install Dependencies and Run Tests:**
   - Install required packages and execute your test suite.

### **Implementation:**

## Continuous Integration

The project uses GitHub Actions for CI. The workflow is defined in `.github/workflows/ci.yml`

1. **Move and Rename the Workflow File:**

   Move your `gitactions` file to `.github/workflows/ci.yml`:

   ```bash
   mkdir -p .github/workflows
   mv .actions/gitactions .github/workflows/ci.yml