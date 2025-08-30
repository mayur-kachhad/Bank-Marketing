# predict.py

import pandas as pd
import joblib
import config

def make_prediction(input_data):
    
    #Load the trained pipeline
    pipeline = joblib.load(config.MODEL_SAVE_PATH)

    #Convert input data to DataFrame
    #crucial because the pipeline expects a DataFrame
    input_df = pd.DataFrame([input_data])
    
    #Get prediction probabilities
    pred_proba = pipeline.predict_proba(input_df)[:, 1][0]

    #Apply the optimal threshold
    prediction = 1 if pred_proba >= config.OPTIMAL_THRESHOLD else 0

    #Determine prediction label
    prediction_label = "Will Subscribe" if prediction == 1 else "Will Not Subscribe"
    
    return {
        "prediction_label": prediction_label,
        "probability": f"{pred_proba:.4f}"
    }