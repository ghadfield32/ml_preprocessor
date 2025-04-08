# DataPreprocessor for Time Series

A flexible and powerful framework to streamline data preprocessing for various machine learning models, including advanced **time series modes** (`set_window`, `pad`, `dtw`) and multiple **splitting methods** (`percentage-based`, `date-based`, `sequence-aware`).

## Table of Contents
1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Time Series Modes](#time-series-modes)
4. [Splitting Methods](#splitting-methods)
5. [Evaluation & Debugging](#evaluation--debugging)
6. [Advanced Usage Example](#advanced-usage-example)
7. [License](#license)

---

## Installation

```bash
# Clone the repo
$ git clone https://github.com/ghadfield32/ml_preprocessor.git
$ cd ml_preprocessor

# Install the package
$ pip install .

# (Optional) Install in editable mode for development
$ pip install -e .

# Install dependencies
$ pip install -r requirements.txt
```

---

## Basic Usage

### 1. Training Mode

```python
import pandas as pd
from datapreprocessor import DataPreprocessor

# Load your dataset
df = pd.read_csv('path_to_your_dataset.csv')

# Initialize DataPreprocessor in train mode
preprocessor = DataPreprocessor(
    model_type="Tree Based Classifier",
    y_variable=["target"],
    ordinal_categoricals=["ordinal_feature1"],
    nominal_categoricals=["nominal_feature1"],
    numericals=["num_feature1"],
    mode="train",
    debug=True
)

# Execute preprocessing
X_train, X_test, y_train, y_test, recommendations, X_test_inverse = \
    preprocessor.final_preprocessing(
        df.drop('target', axis=1),
        df['target']
    )
```

### 2. Prediction Mode

```python
import pandas as pd
from datapreprocessor import DataPreprocessor

# Load new data
new_data = pd.read_csv('new_data.csv')

# Initialize DataPreprocessor in predict mode
predict_preprocessor = DataPreprocessor(
    model_type="Tree Based Classifier",
    y_variable=["target"],
    ordinal_categoricals=["ordinal_feature1"],
    nominal_categoricals=["nominal_feature1"],
    numericals=["num_feature1"],
    mode="predict",
    debug=True
)

# Preprocess new data for predictions
X_preprocessed, recommendations, X_inversed = \
    predict_preprocessor.final_preprocessing(new_data)
```

---

## Time Series Modes

The **DataPreprocessor** supports three key modes for sequence-based data:

1. **set_window**
   ```yaml
   options:
     ts_sequence_mode: "set_window"
     sequence_modes:
       set_window:
         window_size: 10
         max_sequence_length: 10
   ```
   - Defines a fixed window size for each sequence.
   - Ideal for equally spaced time steps.

2. **pad**
   ```yaml
   options:
     ts_sequence_mode: "pad"
     sequence_modes:
       pad:
         pad_threshold: 0.3
         padding_side: "post"
   ```
   - Pads shorter sequences up to a maximum length.
   - Controlled by `pad_threshold` and direction (`pre` or `post`).

3. **dtw**
   ```yaml
   options:
     ts_sequence_mode: "dtw"
     sequence_modes:
       dtw:
         reference_sequence: "max"
         dtw_threshold: 0.3
   ```
   - Uses Dynamic Time Warping for sequence alignment.
   - `reference_sequence` can be a specific ID, `mean`, or `max` length.
   - `dtw_threshold` determines how strictly sequences must match.

---

## Splitting Methods

Depending on your project needs, you can choose from:

1. **Percentage-based Split** (Random split):
   ```yaml
   split_dataset:
     test_size: 0.2
     random_state: 42
   time_series_split:
     method: "standard"
   ```
   - Commonly used (e.g. 80/20 train-test).

2. **Date-based Split**:
   ```yaml
   split_dataset:
     time_split_column: "datetime"
     time_split_value: "2025-02-14 11:50:00"
   time_series_split:
     method: "standard"
   ```
   - Splits by a specific date boundary.

3. **Sequence-aware Split**:
   ```yaml
   time_series_split:
     method: "sequence_aware"
     split_date: "2025-02-14 11:50:00"  # or use target_train_fraction
     target_train_fraction: 0.8
   ```
   - Preserves the integrity of entire sequences for train/test.
   - Useful when you must avoid slicing sequences mid-way.

4. **Feature-Engine or PSI-based Split**:
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
   - Uses PSI-based methods or advanced feature engine logic to determine train/test sets, upgrade over standard percentage-based splits.

---


---

## Advanced Usage Example

Below is a snippet illustrating how you might train and evaluate an LSTM model with **pad** mode and **percentage-based** splitting:

```python
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datapreprocessor import DataPreprocessor

# Load time-series data
data = pd.read_parquet('time_series_data.parquet')

# Configure DataPreprocessor for LSTM, pad mode, and 80/20 percentage split
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

# Analyze potential split points if you want first for specific dates. Percentage based methods are sequence aware
print("Analyzing potential split points...")
split_options = dtw_date_preprocessor.analyze_split_options(data)
for i, option in enumerate(split_options[:3]):  # Show top 3
   print(f"Option {i+1}: Split at {option['split_time']} - Train fraction: {option['train_fraction']:.2f}")

# Preprocess data
X_train, X_test, y_train, y_test, _, _ = preprocessor.final_ts_preprocessing(data)

# Build LSTM model
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)  # horizon = 1
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=32)


```

---

## License

This project is licensed under the [MIT License](LICENSE). Feel free to contribute, modify, and share!

---

