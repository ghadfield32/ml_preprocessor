# clustering_module.py

import logging
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from typing import Optional, Dict
import numpy as np
import pandas as pd

class ClusteringModule:
    def __init__(self, model_type: str = 'K-Means', model_params: Optional[Dict] = None, debug: bool = False):
        """
        Initialize the ClusteringModule with the specified clustering algorithm.

        Args:
            model_type (str): Type of the clustering algorithm ('K-Means', 'DBSCAN', 'AgglomerativeClustering').
            model_params (dict, optional): Parameters for the clustering algorithm.
            debug (bool): Flag to enable detailed debugging.
        """
        self.debug = debug
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = self._initialize_model()
        self.cluster_labels = None
        self.evaluation_metrics = {}
    
    def _initialize_model(self):
        """
        Initialize the clustering algorithm based on the specified type and parameters.

        Returns:
            Clustering algorithm instance.
        """
        try:
            if self.model_type == 'K-Means':
                model = KMeans(**self.model_params)
            elif self.model_type == 'DBSCAN':
                model = DBSCAN(**self.model_params)
            elif self.model_type == 'AgglomerativeClustering':
                model = AgglomerativeClustering(**self.model_params)
            else:
                self.logger.error(f"Unsupported clustering model type: {self.model_type}")
                raise ValueError(f"Unsupported clustering model type: {self.model_type}")
            self.logger.debug(f"Initialized {self.model_type} with parameters: {self.model_params}")
            return model
        except Exception as e:
            self.logger.error(f"Error initializing clustering model: {e}")
            raise e
    
    def fit(self, X: pd.DataFrame):
        """
        Fit the clustering model to the data.

        Args:
            X (pd.DataFrame): Preprocessed feature data.
        """
        try:
            self.logger.info(f"Fitting {self.model_type} model...")
            self.model.fit(X)
            self.cluster_labels = self.model.labels_
            self.logger.info(f"✅ {self.model_type} model fitted successfully.")
        except Exception as e:
            self.logger.error(f"❌ Failed to fit {self.model_type} model: {e}")
            raise e
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Args:
            X (pd.DataFrame): New preprocessed feature data.

        Returns:
            np.ndarray: Cluster labels.
        """
        try:
            self.logger.info(f"Predicting clusters using {self.model_type} model...")
            labels = self.model.predict(X)
            self.logger.info("✅ Clustering predictions made successfully.")
            return labels
        except AttributeError:
            self.logger.error("❌ The clustering model does not support prediction.")
            raise
        except Exception as e:
            self.logger.error(f"❌ Failed to predict clusters: {e}")
            raise e
    
    def evaluate(self, X: pd.DataFrame):
        """
        Evaluate the clustering model using various metrics.

        Args:
            X (pd.DataFrame): Preprocessed feature data.
        """
        try:
            self.logger.info(f"Evaluating {self.model_type} model...")
            self.evaluation_metrics['silhouette_score'] = silhouette_score(X, self.cluster_labels)
            self.evaluation_metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, self.cluster_labels)
            self.evaluation_metrics['davies_bouldin_score'] = davies_bouldin_score(X, self.cluster_labels)
            self.logger.info("✅ Clustering evaluation completed successfully.")
            self.logger.debug(f"Silhouette Score: {self.evaluation_metrics['silhouette_score']:.4f}")
            self.logger.debug(f"Calinski-Harabasz Score: {self.evaluation_metrics['calinski_harabasz_score']:.4f}")
            self.logger.debug(f"Davies-Bouldin Score: {self.evaluation_metrics['davies_bouldin_score']:.4f}")
        except Exception as e:
            self.logger.error(f"❌ Failed to evaluate clustering model: {e}")
            raise e
    
    def plot_clusters(self, X: pd.DataFrame, output_dir: str):
        """
        Plot the clustering results (only for 2D data).

        Args:
            X (pd.DataFrame): Preprocessed feature data.
            output_dir (str): Directory to save the plot.
        """
        try:
            if X.shape[1] != 2:
                self.logger.warning("Plotting is only supported for 2D data.")
                return
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=self.cluster_labels, palette='viridis')
            plt.title(f"{self.model_type} Clustering Results")
            plt.savefig(os.path.join(output_dir, f"{self.model_type}_clusters.png"))
            plt.close()
            self.logger.info(f"✅ Clustering plot saved to '{output_dir}'.")
        except Exception as e:
            self.logger.error(f"❌ Failed to plot clusters: {e}")
            raise e
    
    def save_model(self, filepath: str):
        """
        Save the trained clustering model to disk.

        Args:
            filepath (str): Path to save the model.
        """
        try:
            joblib.dump(self.model, filepath)
            self.logger.info(f"✅ Clustering model saved to '{filepath}'.")
        except Exception as e:
            self.logger.error(f"❌ Failed to save clustering model: {e}")
            raise e
    
    def load_model(self, filepath: str):
        """
        Load a trained clustering model from disk.

        Args:
            filepath (str): Path to load the model from.
        """
        try:
            self.model = joblib.load(filepath)
            self.logger.info(f"✅ Clustering model loaded from '{filepath}'.")
        except Exception as e:
            self.logger.error(f"❌ Failed to load clustering model: {e}")
            raise e
