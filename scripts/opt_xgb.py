import optuna
import pandas as pd
from joblib import dump
from XGBoost import XGBonfolds
from data_cleaning import Data_cleaner


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
        self.df_full=df_full
        self.target_col=target_col
        self.n_folds=n_folds
        self.data_path=data_path
        self.model_path=model_path
        self.study_path=study_path

    def run_optuna(self, n_trials=130):
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
                "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0),
                "enable_categorical":True,
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "num_class": 3,
                "tree_method": "hist",
                #"device": "cuda", # per GPU
                "random_state": 42,
                "verbosity": 0
            }

            model = XGBonfolds(self.df_full, self.data_path, params)
            _, mean_f1= model.run(self.model_path, self.target_col, self.n_folds)

            return mean_f1

        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=20))
        study.optimize(objective, n_trials=n_trials, n_jobs=6, timeout=18000) 

        print("Best F1-micro:", study.best_value)
        print("Best hyperparameters:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        dump(study, self.study_path)
        print(f"Study salvato in: {self.study_path}")

        default_params={
            "enable_categorical":True,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "num_class": 3,
            "tree_method": "hist",
            # "device": "cuda", # per GPU
            "random_state": 42,
            "verbosity": 0
        }
        parameters = {**study.best_params, **default_params}
        save=True
        model = XGBonfolds(self.df_full, self.data_path, parameters)
        f1_scores, mean_f1= model.run(self.model_path, self.target_col, self.n_folds, save)


if __name__ == "__main__":
    dataset_path = 'C:/Users/emagi/Documents/richters_predictor/data/clean_dataset.csv'
    target_col = "damage_grade"
    n_folds = 5
    index_path = "C:/Users/emagi/Documents/richters_predictor/data/cross_validation"
    model_path = "C:/Users/emagi/Documents/richters_predictor/models"
    study_path = "C:/Users/emagi/Documents/richters_predictor/models/optuna_study_xgb.pkl"
    # dataset_emb_path = "C:/Users/emagi/Documents/richters_predictor/data/dataset_with_embedding.csv"      senza embedding

    df = pd.read_csv(dataset_path)
    df = Data_cleaner.missing_and_error_handler(df)    # solo con df senza emb
    print(df.dtypes)
    print(f"Dataset caricato: {df.shape}")
    opt_xgb = XGBoost_tuning(df, target_col, n_folds, index_path, model_path, study_path)
    opt_xgb.run_optuna()
