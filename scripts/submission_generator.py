import pandas as pd
import numpy as np
from joblib import load
from data_cleaning import Data_cleaner
import pickle


class Submitter:
    """
    Classe che crea la submission
    """

    def __init__(self, model_type, test_value_path, submission_format_path, submission_output_path, embedding_dir):
        """
        Inizializza la classe Submitter con i percorsi e le configurazioni per la generazione della submission.

        Parametri:
        - model_type (str): tipo di modello ("xgboost", "catboost", "lightgbm").
        - test_value_path (str): percorso al file CSV con i dati di test.
        - submission_format_path (str): percorso al file CSV con il formato della submission.
        - submission_output_path (str): percorso dove salvare il file CSV di output.
        - embedding_dir (str): directory contenente gli embedding e le mappe ID.

        Ritorna:
        - None
        """
        self.model_type=model_type
        self.test_value_path=test_value_path
        self.submission_format_path=submission_format_path
        self.submission_output_path=submission_output_path
        self.embedding_dir=embedding_dir


    def apply_emb(self, df, column, embedding_matrix, emb_dim, id_map):
        """
        Applica gli embedding a una colonna categoriale del dataframe, mappandone i valori
        a vettori densi tramite una matrice di embedding e un dizionario ID.

        Parametri:
        - df (pd.DataFrame): dataframe a cui applicare gli embedding.
        - column (str): nome della colonna da trasformare.
        - embedding_matrix (np.ndarray): matrice contenente gli embedding.
        - emb_dim (int): dimensione dell'embedding.
        - id_map (dict): dizionario che mappa i valori originali della colonna agli ID di embedding.

        Ritorna:
        - pd.DataFrame: dataframe con i vettori di embedding al posto della colonna originale.
        """
        # Mappa i valori della colonna ai rispettivi ID usando il dizionario fornito
        df[column + "_mapped"] = df[column].map(id_map)

        # Valori non visti nel training
        unseen_values = df[df[column + "_mapped"].isna()][column].unique()
        if len(unseen_values) > 0:
            print(f"{len(unseen_values)} valori NON visti nel training per {column}: {unseen_values}")

        # Embedding dataframe
        emb_df = pd.DataFrame(
            embedding_matrix, 
            columns=[f"{column}_emb_{i}" for i in range(emb_dim)]
        )
        emb_df[column + "_mapped"] = emb_df.index

        # Vettore di default: media degli embedding (indice -1)
        default_vec = embedding_matrix.mean(axis=0)
        default_row = pd.DataFrame([default_vec], columns=[f"{column}_emb_{i}" for i in range(emb_dim)])
        default_row[column + "_mapped"] = -1

        # Aggiunge il vettore di default
        emb_df = pd.concat([emb_df, default_row], ignore_index=True)

        # Sostituisce NaN con -1 e forza a int
        df[column + "_mapped"] = df[column + "_mapped"].astype(float).fillna(-1).astype(int)

        n_default = (df[column + "_mapped"] == -1).sum()
        total = len(df)
        perc = 100 * n_default / total
        print(f"{n_default} campioni test ({perc:.2f}%) usano il vettore di default per {column}")

        # Merge con il DataFrame originale
        df = df.merge(emb_df, on=column + "_mapped", how='left')

        # Elimina la colonna originale e quella mappata
        df = df.drop(columns=[column, column + "_mapped"])

        return df


    def preprocess_test(self, df):
        """
        Applica il preprocessing al dataset di test, rimuovendo colonne non utilizzate,
        gestendo i valori mancanti e applicando gli embedding se richiesto dal tipo di modello.

        Parametri:
        - df (pd.DataFrame): dataset di test grezzo.

        Ritorna:
        - pd.DataFrame: dataset di test preprocessato, pronto per la predizione.
        """
        df = df.drop(columns=["building_id"])
        df = Data_cleaner.missing_and_error_handler(df)

        '''if self.model_type == "xgboost":
            for column in ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]:
                emb_matrix = np.load(f"{self.embedding_dir}/embedding_{column}.npy")
                with open(f"{self.embedding_dir}/id_map_{column}.pkl", "rb") as f:
                    id_map = pickle.load(f)

                emb_dim = emb_matrix.shape[1]
                df = self.apply_emb(df, column, emb_matrix, emb_dim, id_map)'''

        return df


    def generate_submission(self, model_path):
        """
        Genera il file di submission finale combinando le predizioni di ensemble
        su 5 modelli allenati. Salva il risultato nel formato richiesto.

        Parametri:
        - model_path (list of str): sono i path dove salvare i modelli.

        Ritorna:
        - None (ma salva un file CSV di submission nel percorso specificato).
        """
        print(f"Eseguo submission con modello: {self.model_type}")

        df_test = pd.read_csv(self.test_value_path)
        df_test_processed = self.preprocess_test(df_test)

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
        print("Submission creata")


if __name__ == "__main__":
    model_type = "xgboost"  # "lightgbm", "catboost" o "xgboost"
    test_value_path = 'C:/Users/emagi/Documents/richters_predictor/data/test_values.csv'
    submission_format_path = 'C:/Users/emagi/Documents/richters_predictor/data/submission_format.csv'
    submission_output_path = f"C:/Users/emagi/Documents/richters_predictor/data/submission_{model_type}.csv"

    embedding_dir = "C:/Users/emagi/Documents/richters_predictor/data/embeddings"

    model_path = {
            "catboost": [f"C:/Users/emagi/Documents/richters_predictor/models/catboost_model_fold_{i}.joblib" for i in range(1, 6)],
            "lightgbm": [f"C:/Users/emagi/Documents/richters_predictor/models/lightgbm_model_fold_{i}.joblib" for i in range(1, 6)],
            "xgboost": [f"C:/Users/emagi/Documents/richters_predictor/models/xgb_model_fold_{i}.joblib" for i in range(1, 6)],
        }
    
    sub = Submitter (model_type, test_value_path, submission_format_path, submission_output_path, embedding_dir)
    sub.generate_submission(model_path)

