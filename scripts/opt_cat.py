import optuna
# from optuna_integration.catboost import CatBoostPruningCallback
import pandas as pd
from joblib import dump
from data_cleaning import Data_cleaner
from Catboost import CatBoostonfolds


class Catboost_tuning:
    '''
    Classe per eseguire l'ottimizzazione degli iperparametri di un modello CatBoost usando Optuna,
    applicando validazione incrociata sui fold e salvando sia lo studio che i modelli finali.
    
    '''

    def __init__(self, df_full, target_col, n_folds, data_path, model_path, study_path):
        """
        Inizializza la classe CatBoost_tuning con dataset, parametri di tuning e percorsi.

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
        

    def run_optuna(self,n_trials=120):
        """
        Esegue l'ottimizzazione con Optuna per il tuning degli iperparametri
        di un modello CatBoost, basato sulla media del punteggio F1-micro su più fold.

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
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "depth": trial.suggest_int("depth", 5, 10),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
                "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Lossguide"]),
                "loss_function": "MultiClass",
                "eval_metric": "TotalF1:average=Micro",
                "bootstrap_type": "Bayesian",
                "iterations": 1500,
                "one_hot_max_size": 10,
                "early_stopping_rounds": 30,
                "verbose": 0,
                "random_seed": 42,
                "task_type": "CPU"
            }

            model = CatBoostonfolds(self.df_full, self.data_path, params)
            _, mean_f1= model.run(self.model_path, self.target_col, self.n_folds)

            return mean_f1

        # Ottimizzazione
        # Crea uno study Optuna con pruning mediano per velocizzare la ricerca
        study = optuna.create_study(direction="maximize",pruner=optuna.pruners.MedianPruner(n_warmup_steps=20))

        # Avvia l'ottimizzazione sugli iperparametri
        study.optimize(objective, n_trials=n_trials, timeout=30000)

        # Stampa il miglior punteggio e i parametri corrispondenti
        print("Best F1-micro:", study.best_value)
        print("Best hyperparameters:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        # Salva l’oggetto study su disco per usi successivi
        dump(study, self.study_path)
        print(f"Study salvato in: {self.study_path}")

        # Parametri fissi da unire a quelli trovati da Optuna
        default_params={
            "loss_function":"MultiClass",
            "eval_metric":"TotalF1:average=Micro",
            "bootstrap_type":"Bayesian",
            "iterations":1500,
            "early_stopping_rounds":30,
            "verbose":0,
            "random_seed": 42,
            "one_hot_max_size":10,
            "task_type":"CPU",
        }

        # Unione dei parametri 
        parameters = {**study.best_params, **default_params}
        
        # Addestramento finale dei modelli con gli iperparametri ottimizzati e salvataggio su disco
        save=True
        model = CatBoostonfolds(self.df_full, self.data_path, parameters)
        f1_scores, mean_f1= model.run(self.model_path, self.target_col, self.n_folds,save)


if __name__ == "__main__":
    dataset_path = 'C:/Users/emagi/Documents/richters_predictor/data/clean_dataset.csv'
    target_col = "damage_grade"
    n_folds = 5
    index_path = "C:/Users/emagi/Documents/richters_predictor/data/cross_validation"
    model_path = "C:/Users/emagi/Documents/richters_predictor/models"
    study_path = "C:/Users/emagi/Documents/richters_predictor/models/optuna_study_xgb.pkl"

    df = pd.read_csv(dataset_path)
    df = Data_cleaner.missing_and_error_handler(df)   
    print(df.dtypes)
    print(f"Dataset caricato: {df.shape}")
    opt_xgb = Catboost_tuning(df, target_col, n_folds, index_path, model_path, study_path)
    opt_xgb.run_optuna()