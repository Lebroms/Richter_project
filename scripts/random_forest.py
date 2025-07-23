from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

class RFonfolds:
    '''
    Classe per addestrare un classificatore Random Forest su un dataset con validazione incrociata (K-Fold).

    '''
    def __init__(self, df_full, path_dir_csv):
        '''
        Metodo costruttore: inizializza la classe con il dataset e il percorso dei CSV.
         Parametri:
          - df_full (pd.DataFrame): DataFrame contenente il dataset completo.
          - path_dir_csv (str): directory in cui si trovano i file CSV degli indici dei fold.
         Non restituisce nulla.
        '''
        self.df_full = df_full
        self.path_dir_csv = path_dir_csv   

    @staticmethod
    def evaluate_f1_micro(y_true, y_pred):
        '''
        Calcola e stampa l'F1-score micro tra valori veri e predetti.
         Parametri:
          - y_true: (array), valori reali del target.
          - y_pred: (array), valori predetti dal modello.
         Return:
          - f1_micro (float): valore dell'F1-score calcolato con media 'micro'.
        '''
        f1_micro = f1_score(y_true, y_pred, average='micro')
        print(f"F1-micro: {f1_micro:.4f}")
        return f1_micro

    def run(self, target_col='damage_grade', n_folds=5):
        '''
        Esegue la validazione incrociata su n_folds, includendo preprocessing, training e valutazione.
         Parametri:
          - target_col (str): nome della colonna target da predire.
          - n_folds (int): numero di fold da usare nella cross-validation.
         Return:
          - f1_scores (list of float): lista di F1-score per ciascun fold.
          - mean_f1 (float): media degli F1-score sui fold.
        '''

        f1_scores = []

        for fold in range(1, n_folds + 1):
            print(f"Fold {fold}")

            # Legge gli indici di train e validation per il fold corrente
            train_idx = pd.read_csv(f"{self.path_dir_csv}/fold_{fold}_train.csv", header=None)[0].values
            val_idx = pd.read_csv(f"{self.path_dir_csv}/fold_{fold}_val.csv", header=None)[0].values

            # Estrae i subset di train e validation dal dataset originale
            df_train = self.df_full.iloc[train_idx].reset_index(drop=True)
            df_val = self.df_full.iloc[val_idx].reset_index(drop=True)

            # Rimuove colonne geo-level superflue
            df_train = df_train.drop(columns=['geo_level_1_id', 'geo_level_3_id'])
            df_val = df_val.drop(columns=['geo_level_1_id', 'geo_level_3_id'])

            # One-hot encoding per tutte le colonne categoriche
            df_train = pd.get_dummies(df_train, columns=df_train.select_dtypes(include=['object', 'category']).columns)
            df_val = pd.get_dummies(df_val, columns=df_val.select_dtypes(include=['object', 'category']).columns)

            # Allineamento delle colonne tra train e val (in caso di differenze nelle dummies)
            df_train, df_val = df_train.align(df_val, join='left', axis=1, fill_value=0)

            # Divide i dati in feature e target per train e validation
            X_train = df_train.drop(columns=[target_col])
            y_train = df_train[target_col]
            X_val = df_val.drop(columns=[target_col])
            y_val = df_val[target_col]


            model = RandomForestClassifier(
                n_estimators=100,
                max_features='log2',
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Valutazione
            y_pred=model.predict(X_val)
            f1 = self.evaluate_f1_micro(y_val, y_pred)
            f1_scores.append(f1)

            print(f"Fold {fold} completato.")

        # Calcolo della media F1
        mean_f1 = np.mean(f1_scores)
        print(f"F1-micro media su {n_folds} fold: {mean_f1:.4f}")
        return f1_scores, mean_f1