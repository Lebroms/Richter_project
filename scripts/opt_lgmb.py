import optuna
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score
from joblib import dump
from data_cleaning import Data_cleaner
from scripts.LightGBM import LGBMonfolds


class Lgmb_tuning:
    '''
    Classe per eseguire l'ottimizzazione degli iperparametri di un modello LightGBM tramite Optuna.
    Include addestramento e valutazione tramite validazione incrociata su n_folds.
    Attributi:
    - df_full: DataFrame con il dataset completo.
    - target_col: nome della colonna target da predire.
    - n_folds: numero di fold per la validazione incrociata.
    - data_path: percorso alla directory contenente i file CSV degli indici dei fold.
    - model_path: percorso per salvare il modello finale.
    - study_path: percorso per salvare l'oggetto Optuna Study.
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
    def run_optuna(self, n_trials=150):
        '''
        Esegue un'ottimizzazione di iperparametri per un modello LightGBM su n_trials configurazioni.
        Parametri:
        - n_trials: numero di iterazioni da eseguire per la ricerca degli iperparametri.
        Non restituisce nulla esplicitamente (side effect: salva modello e studio su disco).
        '''

        def objective(trial):
            '''
            Funzione obiettivo per Optuna: definisce i parametri da ottimizzare e valuta la media dell'F1-micro score.
            Parametri:
            - trial: oggetto Trial di Optuna che gestisce la proposta della configurazione corrente.
            Return:
            - mean_f1: F1-micro medio sui fold della validazione incrociata per i parametri proposti.
            '''

            

            # Iperparametri suggeriti dinamicamente da Optuna
            params = {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'verbosity': -1,
                'n_estimators': 1500,
                'random_state' : 42,
                'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int("num_leaves", 31, 255),
                'max_depth': trial.suggest_int("max_depth", 4, 15),
                'min_child_samples': trial.suggest_int("min_child_samples", 20, 300),
                'feature_fraction': trial.suggest_float("feature_fraction", 0.6, 1.0),
                'bagging_fraction': trial.suggest_float("bagging_fraction", 0.7, 1.0),
                'bagging_freq': trial.suggest_int("bagging_freq", 1, 5),
                'lambda_l1': trial.suggest_float("lambda_l1", 0.0, 5.0),
                'lambda_l2': trial.suggest_float("lambda_l2", 0.0, 5.0)
            }

            # Inizializzazione e addestramento del modello sui fold specificati
            model = LGBMonfolds(self.df_full,self.data_path,params)
            _, mean_f1= model.run(self.model_path,self.target_col,self.n_folds)

            return mean_f1

        # === Ottimizzazione ===
        # Crea lo studio Optuna, usando MedianPruner per interrompere i trial meno promettenti
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=20))

        # Avvia l'ottimizzazione: usa 6 job paralleli e un timeout massimo di 18000 secondi
        study.optimize(objective, n_trials, n_jobs=6, timeout=18000)

        # Stampa il miglior valore di F1 e i relativi iperparametri
        print("Best F1-micro:", study.best_value)
        print("Best hyperparameters:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        # === Salva lo study ===
        # Serializza e salva l’oggetto study su disco
        dump(study, self.study_path)
        print(f"Study salvato in: {self.study_path}")

        # Parametri fissi di base da unire ai migliori trovati da Optuna
        default_params={
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'verbosity': -1,
            'n_estimators': 1500,
            'random_state' : 42
        }

        # Aggiorna i parametri migliori con quelli fissi (attenzione: update() modifica in-place e restituisce None)
        parameters={**study.best_params,**default_params}

        save=True
        # Addestra il modello finale sui dati completi con i parametri ottimizzati e salva il modello
        model = LGBMonfolds(self.df_full,self.data_path,parameters)
        f1_score, mean_f1= model.run(self.model_path,self.target_col,self.n_folds,save)



if __name__ == "__main__":
    dataset_path = 'C:/Users/emagi/Documents/richters_predictor/data/clean_dataset.csv'
    df = pd.read_csv(dataset_path)
    df = Data_cleaner.missing_and_error_handler(df)
    print(df.dtypes)
    print(f"Dataset caricato: {df.shape}")
    
    ##########AGGIUNGI PARTE DI CHIAMATA DELLA CLASSE