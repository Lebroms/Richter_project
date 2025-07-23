from xgboost import XGBClassifier
# from xgboost.callback import EarlyStopping
from sklearn.metrics import f1_score
from joblib import dump
import pandas as pd
import numpy as np

class XGBonfolds:
    '''
    Classe per addestrare un classificatore XGBoost su un dataset con validazione incrociata (K-Fold).

    '''  
    def __init__(self, df_full, path_dir_csv, params):
        """
        Inizializza la classe XGBonfolds con il dataset completo, il percorso dei file CSV degli indici
        e i parametri dell'XGBoost.

        Parametri:
        - df_full (pd.DataFrame): dataset completo contenente feature e target.
        - path_dir_csv (str): percorso alla directory contenente i CSV con gli indici dei fold.
        - params (dict): dizionario dei parametri da passare a XGBClassifier.

        Ritorna:
        - None
        """
        self.df_full = df_full         
        self.path_dir_csv = path_dir_csv
        self.params = params

    @staticmethod
    def evaluate_f1_micro(y_true, y_pred):
        '''
        Calcola e stampa l'F1-score micro tra valori veri e predetti.
         Parametri:
          - y_true (array): valori reali del target.
          - y_pred (array): valori predetti dal modello.
         Return:
          - f1 (float): valore dell'F1-score calcolato con media 'micro'.
        '''
        f1 = f1_score(y_true, y_pred, average='micro')
        print(f" F1-micro: {f1:.4f}")
        return f1

    def run(self, model_path_dir ,target_col='damage_grade', n_folds=5, save=False):
        """
        Esegue il training e la valutazione di un modello XGBoost su k-fold cross-validation,
        salvando opzionalmente i modelli per ogni fold.

        Parametri:
        - model_path_dir (str): percorso in cui salvare i modelli allenati.
        - target_col (str): nome della colonna target nel dataset.
        - n_folds (int): numero di fold per la cross-validation.
        - save (bool): se True, salva i modelli in formato joblib nella directory specificata.

        Ritorna:
        - f1_scores (list of float): lista dei punteggi F1 per ciascun fold.
        - mean_f1 (float): media dei punteggi F1 su tutti i fold.
        """
        f1_scores = []

        for fold in range(1, n_folds + 1):
            print(f"Fold {fold}")

            # Leggi indici
            train_idx = pd.read_csv(f"{self.path_dir_csv}/fold_{fold}_train.csv", header=None)[0].values
            val_idx = pd.read_csv(f"{self.path_dir_csv}/fold_{fold}_val.csv", header=None)[0].values

            # Converte colonne object in category
            cat_cols = self.df_full.select_dtypes(include=["object"]).columns
            self.df_full[cat_cols] = self.df_full[cat_cols].astype("category")

            # Estrai subset
            df_train = self.df_full.iloc[train_idx].reset_index(drop=True) 
            df_val = self.df_full.iloc[val_idx].reset_index(drop=True)
            
            X_train = df_train.drop(columns=[target_col])
            y_train = df_train[target_col] - 1
            X_val = df_val.drop(columns=[target_col])
            y_val = df_val[target_col] - 1

            model = XGBClassifier(**self.params)
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      # callbacks=[EarlyStopping(rounds=30, save_best=True)],
                      verbose=False)

            if save==True:
                model_path = f"{model_path_dir}/xgb_model_fold_{fold}.joblib"
                dump(model, model_path)

            y_pred = model.predict(X_val)
            f1 = self.evaluate_f1_micro(y_val, y_pred)
            f1_scores.append(f1)

        mean_f1 = np.mean(f1_scores)
        print(f"F1-micro media su {n_folds} fold: {mean_f1:.4f}")
        return f1_scores, mean_f1
    

   
