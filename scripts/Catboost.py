from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from joblib import dump, load
import pandas as pd
import numpy as np
#from optuna_integration.catboost import CatBoostPruningCallback

class CatBoostonfolds:
    '''
    Classe per addestrare un classificatore CatBoost su un dataset con validazione incrociata (K-Fold).
     Attributi:
      - df_full: il dataset completo.
      - path_dir_csv: path della directory contenente i file CSV con gli indici dei fold.
      - params: dizionario dei parametri da passare al modello CatBoost.
      - cat_cols: lista delle colonne categoriali del dataset.
    '''

    def __init__(self, df_full,path_dir_csv,params):
        '''
        Metodo costruttore: inizializza la classe con il dataset, i percorsi e i parametri del modello.
         Parametri:
          - df_full: DataFrame contenente il dataset completo.
          - path_dir_csv: directory in cui si trovano i file CSV degli indici dei fold.
          - params: dizionario dei parametri per CatBoost.
         Non restituisce nulla.
        '''
        self.df_full = df_full

        # Salva il path della directory contenente gli indici dei fold
        self.path_dir_csv = path_dir_csv

        # Salva i parametri del modello
        self.params = params

        # Estrae le colonne categoriali dal dataset
        self.cat_cols=df_full.select_dtypes(include='category').columns.tolist()


    @staticmethod
    def evaluate_f1_micro(y_true, y_pred):
        '''
        Calcola e stampa l'F1-score micro tra valori veri e predetti.
         Parametri:
          - y_true: array-like, valori reali del target.
          - y_pred: array-like, valori predetti dal modello.
         Return:
          - f1: valore dell'F1-score calcolato con media 'micro'.
        '''
        f1 = f1_score(y_true, y_pred, average='micro')
        print(f"F1-micro: {f1:.4f}")
        return f1

    

    def run(self, model_path_dir ,target_col='damage_grade', n_folds=5,save=False):
        '''
        Esegue la validazione incrociata su n_folds, addestra un modello per ogni fold, valuta con F1-score.
         Parametri:
          - model_path_dir: directory in cui salvare i modelli addestrati (se save=True).
          - target_col: nome della colonna target da predire.
          - n_folds: numero di fold da usare nella cross-validation.
          - save: flag booleano che indica se salvare o meno i modelli su disco.
         Return:
          - f1_scores: lista di F1-score per ciascun fold.
          - mean_f1: media degli F1-score sui fold.
        '''
        f1_scores = []

        for fold in range(1, n_folds + 1):
            print(f"\nFold {fold}")

            # Legge gli indici di train e validation per il fold corrente
            train_idx = pd.read_csv(f"{self.path_dir_csv}/fold_{fold}_train.csv", header=None)[0].values
            val_idx = pd.read_csv(f"{self.path_dir_csv}/fold_{fold}_val.csv", header=None)[0].values

            # Estrae i subset di train e validation dal dataset originale
            df_train = self.df_full.iloc[train_idx].reset_index(drop=True)
            df_val = self.df_full.iloc[val_idx].reset_index(drop=True)

            # Divide i dati in feature e target per train e validation
            X_train = df_train.drop(columns=[target_col])
            y_train = df_train[target_col]
            X_val = df_val.drop(columns=[target_col])
            y_val = df_val[target_col]

            # Inizializza il modello CatBoost con i parametri forniti
            model = CatBoostClassifier(**self.params)

            # Addestra il modello sul training set e valuta sul validation set
            model.fit(X_train, y_train,
                      eval_set=(X_val, y_val),
                      cat_features=self.cat_cols)
                      #callbacks=[CatBoostPruningCallback(trial, "TotalF1:average=Micro")]) (solo con GPU)

            if save==True:
                # Salva il modello addestrato del fold corrente su disco
                model_path = f"{model_path_dir}/catboost_model_fold_{fold}.joblib"
                dump(model, model_path)

            # Effettua la predizione sul validation set e valuta con F1 micro
            y_pred = model.predict(X_val)
            f1 = self.evaluate_f1_micro(y_val, y_pred)
            f1_scores.append(f1)

        # Calcola la media degli F1-score sui fold
        mean_f1 = np.mean(f1_scores)
        print(f"\n📊 F1-micro media su {n_folds} fold: {mean_f1:.4f}")
        return f1_scores, mean_f1
