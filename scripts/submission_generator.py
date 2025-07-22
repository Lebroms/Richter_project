import pandas as pd
import numpy as np
from joblib import load
from data_cleaning import Data_cleaner

# === CONFIGURAZIONE ===
MODEL_TYPE = "lightgbm"  # "lightgbm", "catboost" o "xgboost"

TEST_VALUES_PATH = 'C:/Users/emagi/Documents/richters_predictor/data/test_values.csv'
SUBMISSION_FORMAT_PATH = 'C:/Users/emagi/Documents/richters_predictor/data/submission_format.csv'
SUBMISSION_OUTPUT_PATH = f"C:/Users/emagi/Documents/richters_predictor/data/submission_{MODEL_TYPE}.csv"

MODEL_PATHS = {
    "catboost": [f"C:/Users/emagi/Documents/richters_predictor/models/catboost_model_fold_{i}.joblib" for i in range(1, 6)],
    "lightgbm": [f"C:/Users/emagi/Documents/richters_predictor/models/lightgbm_model_fold_{i}.joblib" for i in range(1, 6)],
    "xgboost": [f"C:/Users/emagi/Documents/richters_predictor/models/xgboost_model_fold_{i}.joblib" for i in range(1, 6)],
}

# === FUNZIONI ===

def preprocess_test(df, model_type):
    df = df.drop(columns=["building_id"])
    if model_type in ["catboost", "lightgbm"]:
        df = Data_cleaner.missing_and_error_handler(df)
    elif model_type == "xgboost":
        # Sostituisci questo con la tua funzione di embedding reale
        df["geo_level_3_id"] = df["geo_level_3_id"].astype('category')
        df = pd.get_dummies(df)
    return df

# === MAIN ===
if __name__ == "__main__":
    print(f"🔍 Eseguo submission con modello: {MODEL_TYPE}")

    df_test = pd.read_csv(TEST_VALUES_PATH)
    df_test_processed = preprocess_test(df_test.copy(), MODEL_TYPE)

    all_preds = []

    for model_path in MODEL_PATHS[MODEL_TYPE]:
        model = load(model_path)
        probs = model.predict_proba(df_test_processed)
        all_preds.append(probs)

    # Media delle probabilità
    avg_probs = np.mean(all_preds, axis=0)
    final_preds = np.argmax(avg_probs, axis=1)

    submission_df = pd.read_csv(SUBMISSION_FORMAT_PATH)
    submission_df["damage_grade"] = final_preds
    submission_df.to_csv(SUBMISSION_OUTPUT_PATH, index=False)
    print(f"Submission salvata in: {SUBMISSION_OUTPUT_PATH}")



 

