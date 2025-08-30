# train_pipeline.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier
import joblib
import config  

def run_training():
    """Trains and saves the model pipeline."""
    
    #Load Data
    df = pd.read_csv(config.DATA_FILE_PATH)
    
    #Map Target Variable
    df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].map({'yes': 1, 'no': 0})
    
    #Features (X) and Target (y)
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]

    #Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), config.NUM_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), config.CAT_COLS)
        ],
        remainder='passthrough'
    )
    
    #Full ML Pipeline
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier(**config.LGBM_BEST_PARAMS))
    ])
    
    #Training the Pipeline
    print("Training the pipeline on the full dataset...")
    final_pipeline.fit(X, y)
    print("Training complete.")

    #Save Pipeline
    joblib.dump(final_pipeline, config.MODEL_SAVE_PATH)
    print(f"Pipeline saved to: {config.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    run_training()