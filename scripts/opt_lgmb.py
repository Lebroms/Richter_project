import optuna
import pandas as pd
from joblib import dump
from data_cleaning import Data_cleaner
from LightGBM import LGBMonfolds


class Lgmb_tuning:
    '''
    Classe per eseguire l'ottimizzazione degli iperparametri di un modello LightGBM usando Optuna,
    applicando validazione incrociata sui fold e salvando sia lo studio che i modelli finali.
    
    '''

    def __init__(self,df_full,target_col,n_folds,data_path,model_path,study_path):
        """
        Inizializza la classe Lgmb_tuning con dataset, parametri di tuning e percorsi.

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
        
    def run_optuna(self, n_trials=150):
        """
        Esegue l'ottimizzazione con Optuna per il tuning degli iperparametri
        di un modello LightGBM, basato sulla media del punteggio F1-micro su più fold.

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

        # Ottimizzazione
        # Crea lo studio Optuna, usando MedianPruner per interrompere i trial meno promettenti
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=20))

        # Avvia l'ottimizzazione
        study.optimize(objective, n_trials, n_jobs=6, timeout=18000)

        # Stampa il miglior valore di F1 e i relativi iperparametri
        print("Best F1-micro:", study.best_value)
        print("Best hyperparameters:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        # salva l’oggetto study su disco
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

        # Aggiorna i parametri migliori con quelli fissi 
        parameters={**study.best_params,**default_params}

        save=True
        # Addestra i modelli finali con gli iperparametri ottimizzati e salva i modelli
        model = LGBMonfolds(self.df_full, self.data_path, parameters)
        f1_score, mean_f1= model.run(self.model_path, self.target_col, self.n_folds, save)



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
    opt_xgb = Lgmb_tuning(df, target_col, n_folds, index_path, model_path, study_path)
    opt_xgb.run_optuna()