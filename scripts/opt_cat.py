import optuna

import pandas as pd
import numpy as np

from joblib import dump
from data_cleaning import Data_cleaner
from scripts.Catboost import CatBoostonfolds


class Catboost_tuning:
    '''
    Classe per eseguire l'ottimizzazione degli iperparametri di un modello CatBoost usando Optuna,
    applicando validazione incrociata sui fold e salvando sia lo studio che il modello finale.
    Attributi:
    - df_full: DataFrame contenente il dataset completo.
    - target_col: nome della colonna target da predire.
    - n_folds: numero di fold per la cross-validation.
    - data_path: percorso della directory con i file CSV degli indici.
    - model_path: percorso per il salvataggio del modello finale.
    - study_path: percorso per il salvataggio dell’oggetto Optuna study.
    '''

    def __init__(self,df_full,target_col,n_folds,data_path,model_path,study_path):
        '''
        Metodo costruttore: inizializza gli attributi della classe.
        Parametri:
        - df_full: DataFrame contenente il dataset completo.
        - target_col: nome della colonna target.
        - n_folds: numero di fold per la validazione incrociata.
        - data_path: directory contenente i CSV degli indici di fold.
        - model_path: path in cui salvare il modello finale.
        - study_path: path in cui salvare lo studio Optuna.
        Non restituisce nulla.
        '''
        self.df_full=df_full
        self.target_col=target_col
        self.n_folds=n_folds
        self.data_path=data_path
        self.model_path=model_path
        self.study_path=study_path
        

    

    # === Funzione principale ===
    def run_optuna(self,n_trials=120):
        '''
        Funzione per ottimizzare gli iperparametri del modello CatBoost con Optuna e addestrare il modello finale con i parametri ottimali.
        Parametri:
        - n_trials: numero di tentativi di ricerca iperparametri da effettuare.
        Non restituisce nulla esplicitamente (side effect: salva modello e studio su disco).
        '''

        def objective(trial):
            '''
            Funzione obiettivo per la ricerca degli iperparametri con Optuna.
            Parametri:
            - trial: oggetto Optuna che gestisce la proposta di combinazioni iperparametriche.
            Return:
            - mean_f1: punteggio F1 micro medio sui fold per la configurazione testata.
            '''

            # Iperparametri suggeriti dinamicamente da Optuna
            params = {
                "iteriations":trial.suggest_int("iterations", 300, 2000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "depth": trial.suggest_int("depth", 5, 10),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
                "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Lossguide"]),
                "loss_function": "MultiClass",
                "eval_metric": "TotalF1:average=Micro",
                "bootstrap_type": "Bayesian",
                # "iterations": 1500,   solo con early_stopping_rounds
                "one_hot_max_size": 10,
                # "early_stopping_rounds": 30,
                "verbose": 0,
                "random_seed": 42,
                "task_type": "CPU"
            }

            # Inizializza il modello CatBoost per la cross-validation su più fold
            model = CatBoostonfolds(self.df_full,self.data_path,params)
            _, mean_f1= model.run(self.model_path,self.target_col,self.n_folds)

            return mean_f1

        # === Ottimizzazione ===
        # Crea uno study Optuna con pruning mediano per velocizzare la ricerca
        study = optuna.create_study(direction="maximize",pruner=optuna.pruners.MedianPruner(n_warmup_steps=20))

        # Avvia l'ottimizzazione sugli iperparametri
        study.optimize(objective, n_trials=n_trials, timeout=30000) #n_jobs=6 (solo con CPU)

        # Stampa il miglior punteggio e i parametri corrispondenti
        print("Best F1-micro:", study.best_value)
        print("Best hyperparameters:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        # === Salva lo study ===
        # Salva l’oggetto study su disco per usi successivi
        dump(study, self.study_path)
        print(f"Study salvato in: {self.study_path}")

        # Parametri fissi da unire a quelli trovati da Optuna
        default_params={
            "loss_function":"MultiClass",
            "eval_metric":"TotalF1:average=Micro",
            "bootstrap_type":"Bayesian",
            #"iterations":1500,
            #"early_stopping_rounds":30,
            "verbose":0,
            "random_seed": 42,
            "one_hot_max_size":10,
            "task_type":"CPU",
        }

        # Unione dei parametri (attenzione: update modifica in-place e ritorna None)
        parameters={**study.best_params,**default_params}

        save=True
        # Addestramento finale del modello con i parametri ottimizzati e salvataggio su disco
        model = CatBoostonfolds(self.df_full,self.data_path,parameters)
        f1_scores, mean_f1= model.run(self.model_path,self.target_col,self.n_folds,save)


if __name__ == "__main__":
    dataset_path = 'C:/Users/emagi/Documents/richters_predictor/data/clean_dataset.csv'
    df = pd.read_csv(dataset_path)
    df = Data_cleaner.missing_and_error_handler(df)
    print(df.dtypes)
    print(f"Dataset caricato: {df.shape}")
    
    ##########AGGIUNGI PARTE DI CHIAMATA DELLA CLASSE