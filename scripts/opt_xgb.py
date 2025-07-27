import optuna
import pandas as pd
import numpy as np
import pickle
from joblib import dump
from XGBoost import XGBonfolds
from data_cleaning import Data_cleaner

BASE_PATH = 'C:/Users/lscor/OneDrive/Magistrale/F_Intelligenza_artificiale/Richter_project/data/embeddings/'

def load_embedding_and_map(column):
    """
    Carica da file la matrice di embedding e la mappa ID associata a una colonna categoriale.

    Parametri:
    - column (str): nome della colonna per cui caricare l'embedding 

    Ritorna:
    - tuple (np.ndarray, dict): 
        - emb_matrix: matrice NumPy con i vettori di embedding.
        - id_map: dizionario che mappa i valori originali della colonna agli ID usati per indexing.
    """
    emb_matrix = np.load(f"{BASE_PATH}/embedding_{column}.npy")
    with open(f"{BASE_PATH}/id_map_{column}.pkl", "rb") as f:
        id_map = pickle.load(f)
    return emb_matrix, id_map



class XGBoost_tuning:
    """
    Classe per eseguire l'ottimizzazione degli iperparametri di un modello XGBoost usando Optuna,
    applicando validazione incrociata sui fold e salvando sia lo studio che i modelli finali.

    """
    def __init__(self, df_full, target_col, n_folds, data_path, model_path, study_path):
        """
        Inizializza la classe XGBoost_tuning con dataset, parametri di tuning e percorsi.

        Parametri:
        - df_full (pd.DataFrame): dataset completo contenente le feature e il target.
        - target_col (str): nome della colonna target.
        - n_folds (int): numero di fold per la cross-validation.
        - data_path (str): percorso della directory contenente gli indici dei fold.
        - model_path (str): percorso della directory dove salvare i modelli allenati.
        - study_path (str): percorso del file in cui salvare l'oggetto Optuna `study`.

        Ritorna:
        - None
        """
        self.df_full = df_full
        self.target_col = target_col
        self.n_folds = n_folds
        self.data_path = data_path
        self.model_path = model_path
        self.study_path = study_path
        self.embedding_columns = ["geo_level_2_id", "geo_level_3_id"]
        self.embedding_data = {
            col: load_embedding_and_map(col) for col in self.embedding_columns
        }

    def run_optuna(self, n_trials=200):
        """
        Esegue l'ottimizzazione con Optuna per il tuning degli iperparametri
        di un modello XGBoost, basato sulla media del punteggio F1-micro su più fold.

        Parametri:
        - n_trials (int): numero massimo di tentativi per Optuna.

        Ritorna:
        - None (ma stampa i migliori iperparametri e salva i modelli finali e lo `study`).

        """
        def objective(trial):
            """
            Funzione obiettivo per Optuna, che esegue il training
            un set di iperparametri suggeriti e restituisce il F1-micro medio.

            Parametri:
            - trial (optuna.trial.Trial): oggetto trial che fornisce i suggerimenti.

            Ritorna:
            - float: punteggio F1-micro medio sui fold.
            """
            params = {
                #"n_estimators": trial.suggest_int("n_estimators", 300, 2000),  # accorciato range
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 4, 15),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 8.0),
                "gamma": trial.suggest_float("gamma", 0, 3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 2.0),
                "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
                "n_estimators": 2000,
                "enable_categorical": True,
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "num_class": 3,
                "tree_method": "hist",
                "random_state": 42,
                "verbosity": 0
            }



            # Apply fold embeddings dentro il modello
            model = XGBonfolds(self.df_full, self.data_path, params, self.embedding_data)
            _, mean_f1 = model.run(self.model_path, self.target_col, self.n_folds,trial_number=trial.number)

            return mean_f1

        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=30))
        study.optimize(objective, n_trials=n_trials, n_jobs=6, timeout=27000)

        print("Best F1-micro:", study.best_value)
        print("Best hyperparameters:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        dump(study, self.study_path)
        print(f"Study salvato in: {self.study_path}") 

        default_params = {
            "n_estimators": 2000,
            "enable_categorical": True,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "num_class": 3,
            "tree_method": "hist",
            "random_state": 42,
            "verbosity": 0
        }
        parameters = {**study.best_params, **default_params}
        model = XGBonfolds(self.df_full, self.data_path, parameters, self.embedding_data)
        model.run(self.model_path, self.target_col, self.n_folds, save=True)

if __name__ == "__main__":
    dataset_path = 'C:/Users/lscor/OneDrive/Magistrale/F_Intelligenza_artificiale/Richter_project/data/clean_dataset.csv'
    target_col = "damage_grade"
    n_folds = 5
    index_path = "C:/Users/lscor/OneDrive/Magistrale/F_Intelligenza_artificiale/Richter_project/data/cross_validation/"
    model_path = "C:/Users/lscor/OneDrive/Magistrale/F_Intelligenza_artificiale/Richter_project/models/"
    study_path = "C:/Users/lscor/OneDrive/Magistrale/F_Intelligenza_artificiale/Richter_project/models/optuna_study_xgb.pkl"

    df = pd.read_csv(dataset_path)
    df = Data_cleaner.missing_and_error_handler(df)
    print(df.dtypes)
    print(f"Dataset caricato: {df.shape}")
    tuner = XGBoost_tuning(df, target_col, n_folds, index_path, model_path, study_path)
    tuner.run_optuna()
