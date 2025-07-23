import lightgbm as lgb
from sklearn.metrics import f1_score
from joblib import dump
import pandas as pd
from data_cleaning import Data_cleaner
import numpy as np

class LGBMonfolds:
    '''
    Classe per addestrare un classificatore LightGBM su un dataset con validazione incrociata (K-Fold).

    '''

    def __init__(self, df_full,path_dir_csv,params):
        '''
        Metodo costruttore: inizializza la classe con il dataset, i percorsi e i parametri del modello.

        Parametri:
        - df_full (pd.DataFrame): dataset completo contenente feature e target.
        - path_dir_csv (str): percorso alla directory contenente i CSV con gli indici dei fold.
        - params (dict): dizionario dei parametri da passare a LGBMClassifier.

        Ritorna:
        - None
        '''

        self.df_full = df_full

        # Salva il path della directory contenente gli indici dei fold
        self.path_dir_csv = path_dir_csv

        # Salva i parametri del modello
        self.params = params

        # Estrae e salva la lista delle colonne categoriali
        self.cat_cols=df_full.select_dtypes(include='category').columns.tolist()

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
        print(f"F1-micro: {f1:.4f}")
        return f1
    
    def run(self, model_path_dir,target_col='damage_grade', n_folds=5,save=False):
        """
        Esegue il training e la valutazione di un modello LightGBM su k-fold cross-validation,
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
            
            # Caricamento degli indici di train e validation dai file CSV per il fold corrente
            train_idx = pd.read_csv(f"{self.path_dir_csv}/fold_{fold}_train.csv", header=None)[0].values
            val_idx = pd.read_csv(f"{self.path_dir_csv}/fold_{fold}_val.csv", header=None)[0].values

            # Estrazione dei sotto-dataframe per training e validation
            df_train = self.df_full.iloc[train_idx].reset_index(drop=True)
            df_val = self.df_full.iloc[val_idx].reset_index(drop=True)

            # Separazione delle feature e del target per il training e la validazione
            X_train = df_train.drop(columns=[target_col])
            y_train = df_train[target_col]
            X_val = df_val.drop(columns=[target_col])
            y_val = df_val[target_col]

            # Inizializzazione del modello LightGBM con i parametri forniti
            model = lgb.LGBMClassifier(**self.params)
            
            # Addestramento del modello sul training set con early stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='multi_logloss',
                callbacks=[lgb.early_stopping(stopping_rounds=40)]
            )

            if save==True:
                # Salvataggio del modello addestrato su disco
                model_path = f"{model_path_dir}/lightgbm_model_fold_{fold}.joblib"
                dump(model, model_path)

            # Predizione sul validation set e calcolo dell'F1-score micro
            y_pred = model.predict(X_val)
            f1 = self.evaluate_f1_micro(y_val, y_pred)
            f1_scores.append(f1)
        
        # Calcolo della media degli F1-score sui fold
        mean_f1 = np.mean(f1_scores)
        print(f"F1-micro media su {n_folds} fold: {mean_f1:.4f}")
        return f1_scores, mean_f1


