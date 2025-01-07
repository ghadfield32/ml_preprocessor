# feature_manager.py

import logging
import pandas as pd
import pickle
from typing import Optional, List, Dict, Any, Tuple
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more verbose output
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('feature_manager')


class FeatureManager:
    def __init__(self, save_path: str = 'features_metadata.pkl'):
        self.save_path = save_path
    
    def save_features(
        self,
        features_df: pd.DataFrame,
        ordinal_categoricals: Optional[List[str]],
        nominal_categoricals: Optional[List[str]],
        numericals: Optional[List[str]],
        y_variable: List[str],
        dataset_csv_path: str
    ) -> None:
        """
        Save selected features and their metadata to a pickle file.
        """
        try:
            if features_df.empty:
                raise ValueError("features_df is empty. Cannot save empty features.")
            if not os.path.exists(dataset_csv_path):
                raise FileNotFoundError(f"Dataset CSV file does not exist at path: {dataset_csv_path}")
            
            # Prepare metadata
            data_to_save = {
                'features': features_df.columns.tolist(),
                'ordinal_categoricals': ordinal_categoricals or [],
                'nominal_categoricals': nominal_categoricals or [],
                'numericals': numericals or [],
                'y_variable': y_variable,
                'dataset_csv_path': dataset_csv_path
            }
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            
            # Save metadata to pickle
            with open(self.save_path, 'wb') as f:
                pickle.dump(data_to_save, f)
            logger.info(f"✅ Features and metadata saved to {self.save_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save features and metadata: {e}")
            raise  # Re-raise exception after logging
    
    def load_features_and_dataset(
        self,
        debug: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """
        Load features and metadata from a pickle file, then load and filter the original dataset based on the loaded features.
        """
        try:
            if not os.path.exists(self.save_path):
                raise FileNotFoundError(f"The file {self.save_path} does not exist.")
            
            # Load metadata from pickle
            with open(self.save_path, 'rb') as f:
                loaded_data = pickle.load(f)
            logger.info(f"✅ Features and metadata loaded from {self.save_path}")
            
            # Extract metadata
            selected_features = loaded_data.get('features', [])
            ordinal_categoricals = loaded_data.get('ordinal_categoricals', [])
            nominal_categoricals = loaded_data.get('nominal_categoricals', [])
            numericals = loaded_data.get('numericals', [])
            y_variable = loaded_data.get('y_variable', [])
            dataset_path = loaded_data.get('dataset_csv_path', '')
            
            logger.debug(f"Dataset Path Retrieved: {dataset_path}")
            logger.debug(f"Number of Features Selected: {len(selected_features)}")
            logger.debug(f"Features Selected: {selected_features}")
            
            # Validate dataset_path
            if not dataset_path:
                logger.error("Dataset path is empty in the loaded metadata.")
                raise ValueError("Dataset path is not provided in the loaded metadata.")
            
            if not os.path.exists(dataset_path):
                logger.error(f"Dataset CSV file does not exist at path: {dataset_path}")
                raise FileNotFoundError(f"Dataset CSV file not found at {dataset_path}")
            
            # Load the original dataset
            logger.info(f"📥 Loading dataset from {dataset_path}...")
            original_df = load_base_data_for_dataset(dataset_path)
            logger.info("✅ Original dataset loaded successfully.")
            
            # Filter the dataset based on selected features
            logger.info("🔍 Filtering dataset for selected features...")
            filtered_df = filter_base_data_for_select_features(
                dataset=original_df,
                feature_names=selected_features,
                debug=debug
            )
            logger.info("✅ Dataset filtered successfully.")
            
            # Separate column assets
            logger.info("📁 Separating columns into defined categories...")
            column_assets = separate_column_assets(
                feature_names=selected_features,
                ordinal_categoricals=ordinal_categoricals,
                nominal_categoricals=nominal_categoricals,
                numericals=numericals,
                y_variable=y_variable
            )
            logger.debug(f"Column Assets Separated: {column_assets}")
            
            logger.info("✅ Features loaded and dataset filtered successfully.")
            return filtered_df, column_assets
        
        except Exception as e:
            logger.error(f"❌ Failed to load features and dataset: {e}")
            raise  # Re-raise exception after logging


def load_base_data_for_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Args:
        dataset_path (str): Path to the dataset CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        df = pd.read_csv(dataset_path)
        logger.info(f"✅ Dataset loaded from {dataset_path}")
        return df
    except Exception as e:
        logger.error(f"❌ Failed to load dataset from {dataset_path}: {e}")
        raise

def filter_base_data_for_select_features(dataset: pd.DataFrame, feature_names: List[str], debug: bool = False) -> pd.DataFrame:
    """
    Filter the dataset to include only the selected features.

    Args:
        dataset (pd.DataFrame): The original dataset.
        feature_names (List[str]): List of feature names to retain.
        debug (bool): Flag to enable debug logging.

    Returns:
        pd.DataFrame: Filtered dataset.
    """
    try:
        filtered_df = dataset[feature_names].copy()
        if debug:
            logger.debug(f"Filtered dataset shape: {filtered_df.shape}")
        logger.info("✅ Dataset filtered based on selected features.")
        return filtered_df
    except KeyError as e:
        logger.error(f"❌ One or more features not found in the dataset: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Failed to filter dataset: {e}")
        raise

def separate_column_assets(
    feature_names: List[str],
    ordinal_categoricals: List[str],
    nominal_categoricals: List[str],
    numericals: List[str],
    y_variable: List[str]
) -> Dict[str, List[str]]:
    """
    Separate feature names into their respective categories.

    Args:
        feature_names (List[str]): List of feature names.
        ordinal_categoricals (List[str]): List of ordinal categorical feature names.
        nominal_categoricals (List[str]): List of nominal categorical feature names.
        numericals (List[str]): List of numerical feature names.
        y_variable (List[str]): List containing the target variable name.

    Returns:
        Dict[str, List[str]]: Dictionary containing separated feature categories.
    """
    try:
        remaining_features = set(feature_names) - set(ordinal_categoricals) - set(nominal_categoricals) - set(numericals) - set(y_variable)
        if remaining_features:
            logger.warning(f"⚠️ The following features were not categorized: {remaining_features}")
        
        column_assets = {
            'ordinal_categoricals': ordinal_categoricals,
            'nominal_categoricals': nominal_categoricals,
            'numericals': numericals,
            'y_variable': y_variable
        }
        logger.info("✅ Columns separated into defined categories.")
        return column_assets
    except Exception as e:
        logger.error(f"❌ Failed to separate column assets: {e}")
        raise



# example of feature manager usages

import logging
import pandas as pd
import os
# from feature_manager import FeatureManager

# Configure logging for this script
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more verbose output
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('example_usage_class')

def main():
    try:
        # Initialize FeatureManager with the desired save_path
        save_path = '../../dataset/test/features_info/features_metadata.pkl'  # Pickle file path
        feature_manager = FeatureManager(save_path=save_path)
        
        start_dataset_path = '../../dataset/test/data/final_ml_dataset.csv'  # Original dataset CSV path
        
        # Check if the original dataset exists
        if not os.path.exists(start_dataset_path):
            logger.error(f"Original dataset does not exist at path: {start_dataset_path}")
            raise FileNotFoundError(f"Original dataset not found at {start_dataset_path}")
        
        # Load the original dataset
        logger.info(f"📥 Loading original dataset from {start_dataset_path}...")
        original_df = pd.read_csv(start_dataset_path)
        logger.info("✅ Original dataset loaded successfully.")
        
        # Define feature categories and column names
        ordinal_categoricals = []
        nominal_categoricals = []
        numericals = [
            'release_ball_direction_x', 'release_ball_direction_z', 'release_ball_direction_y',
            'elbow_release_angle', 'elbow_max_angle',
            'wrist_release_angle', 'wrist_max_angle',
            'knee_release_angle', 'knee_max_angle',
             'release_ball_speed', 'calculated_release_angle',
            'release_ball_velocity_x', 'release_ball_velocity_y', 'release_ball_velocity_z'
        ]
        y_variable = ['result']
        final_keep_list = ordinal_categoricals + nominal_categoricals + numericals + y_variable
        
        # Verify that all columns in final_keep_list exist in the dataset
        missing_columns = set(final_keep_list) - set(original_df.columns)
        if missing_columns:
            logger.error(f"The following columns are missing in the dataset: {missing_columns}")
            raise ValueError(f"Missing columns in the dataset: {missing_columns}")
        
        # Apply the filter to keep only the selected columns
        logger.info("🔍 Selecting and filtering dataset based on defined features...")
        selected_features_df = original_df[final_keep_list]
        logger.info("✅ Selected features filtered successfully.")
        
        # Save features and metadata
        try:
            feature_manager.save_features(
                features_df=selected_features_df,
                ordinal_categoricals=ordinal_categoricals,
                nominal_categoricals=nominal_categoricals,
                numericals=numericals,
                y_variable=y_variable,
                dataset_csv_path=start_dataset_path
            )
        except Exception as e:
            logger.error(f"❌ Failed to save features and metadata: {e}")
            raise
        
        # Load features and dataset
        try:
            filtered_df, column_assets = feature_manager.load_features_and_dataset(
                debug=True  # Set to False to reduce verbosity
            )
            logger.info("✅ Features loaded and dataset filtered successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to load features and dataset: {e}")
            raise
        
        # Display processed data
        logger.info("\n📊 Processed DataFrame (first 5 rows):")
        logger.info(f"{filtered_df.head()}")
        
        # Display separated column assets
        logger.info("\n📁 Separated Column Assets:")
        for key, value in column_assets.items():
            logger.info(f"{key}: {value}")
        
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    main()

