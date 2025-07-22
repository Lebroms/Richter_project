# submission_generator.py
import pandas as pd
import numpy as np
from joblib import load
from data_cleaning import Data_cleaner
import pickle



class Submitter:

    def __init__(self,model_type,test_value_path,submission_format_path,submission_output_path,embedding_dir):
        self.model_type=model_type
        self.test_value_path=test_value_path
        self.submission_format_path=submission_format_path
        self.submission_output_path=submission_output_path
        self.embedding_dir=embedding_dir

    
    
    
    def apply_embedding(self,df, column, embedding_matrix, emb_dim, id_map):
        # Mappa i valori della colonna ai rispettivi ID usando il dizionario fornito
        df[column + "_mapped"] = df[column].map(id_map)

        # Genera DataFrame dell'embedding
        emb_df = pd.DataFrame(
            embedding_matrix, 
            columns=[f"{column}_emb_{i}" for i in range(emb_dim)]
        )
        emb_df[column + "_mapped"] = emb_df.index

        # Merge con il DataFrame originale
        df = df.merge(emb_df, on=column + "_mapped", how='left')

        # Elimina colonna originale e la colonna mapped
        df = df.drop(columns=[column, column + "_mapped"])
        
        return df
    


    def preprocess_test(self,df):
        df = df.drop(columns=["building_id"])
        df = Data_cleaner.missing_and_error_handler(df)

        if self.model_type == "xgboost":
            for column in ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]:
                emb_matrix = np.load(f"{self.embedding_dir}/embedding_{column}.npy")
                with open(f"{self.embedding_dir}/id_map_{column}.pkl", "rb") as f:
                    id_map = pickle.load(f)

                emb_dim = emb_matrix.shape[1]
                df = self.apply_embedding(df, column, emb_matrix, emb_dim, id_map)

        return df
    

    def generate_submission(self):
        df_test = pd.read_csv(self.test_value_path)
        df_test_processed = self.preprocess_test(df_test, self.model_type)

        model_path = {
            "catboost": [f"C:/Users/emagi/Documents/richters_predictor/models/catboost_model_fold_{i}.joblib" for i in range(1, 6)],
            "lightgbm": [f"C:/Users/emagi/Documents/richters_predictor/models/lightgbm_model_fold_{i}.joblib" for i in range(1, 6)],
            "xgboost": [f"C:/Users/emagi/Documents/richters_predictor/models/xgb_model_fold_{i}.joblib" for i in range(1, 6)],
        }

        all_preds = []
        for path in model_path[self.model_type]:
            model = load(path)
            probs = model.predict_proba(df_test_processed)
            all_preds.append(probs)


        # Media delle probabilità
        avg_probs = np.mean(all_preds, axis=0)
        final_preds = np.argmax(avg_probs, axis=1)

        submission_df = pd.read_csv(self.submission_format_path)
        submission_df["damage_grade"] = final_preds + 1
        submission_df.to_csv(self.submission_output_path, index=False)
        


        










 

