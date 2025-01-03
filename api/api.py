from fastapi import FastAPI, UploadFile
import pandas as pd
from datapreprocessor import DataPreprocessor
from datapreprocessor.utils.yaml_utils import load_config

app = FastAPI()

# Load config once at startup
config = load_config("datapreprocessor/config/preprocessor_config.yaml")

@app.post("/train")
async def train(file: UploadFile):
    df = pd.read_csv(file.file)
    # Example: run preprocessor in train mode
    preprocessor = DataPreprocessor(
        model_type=config["current_model"],
        column_assets=config["features"],
        mode="train",
        options=config["models"][config["current_model"]],
    )
    X_train, X_test, y_train, y_test, recommendations, X_test_inverse = preprocessor.final_preprocessing(df)
    return {
        "X_train_shape": X_train.shape,
        "recommendations_head": recommendations.head().to_dict(orient="records"),
    }
