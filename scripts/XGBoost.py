from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from joblib import dump
import pandas as pd
import numpy as np

def apply_fold_embedding(df_train, df_val, column, emb_matrix, id_map):

    """
    Applica l'embedding a una colonna categoriale sia sul training set che sul validation set,
    mappando i valori tramite id_map e assegnando i vettori da emb_matrix. 
    Per i valori unseen nel validation set, assegna un vettore medio ponderato calcolato sul training.

    Parametri:
    - df_train (pd.DataFrame): DataFrame contenente il training set dell'esperimento.
    - df_val (pd.DataFrame): DataFrame contenente il fold di validazione dell'esperimento.
    - column (str): nome della colonna categoriale su cui applicare l'embedding.
    - emb_matrix (np.ndarray): matrice di embedding, con un vettore per ciascun ID mappato.
    - id_map (dict): dizionario che mappa i valori della colonna ai rispettivi ID numerici.

    Ritorna:
    - tuple (pd.DataFrame, pd.DataFrame): DataFrame di training e validation arricchiti con le colonne
      di embedding e privi della colonna originale e della colonna mappata.
    """
        
    emb_dim = emb_matrix.shape[1]

    # Mappa i valori del training set 
    df_train[column + "_mapped"] = df_train[column].map(id_map).astype(float).fillna(-1).astype(int)

    # Costruisce DataFrame con gli embedding usati nel training 
    present_ids = df_train[column + "_mapped"].unique()
    emb_df = pd.DataFrame(emb_matrix, columns=[f"{column}_emb_{i}" for i in range(emb_dim)])
    emb_df["id"] = emb_df.index
    emb_df = emb_df[emb_df["id"].isin(present_ids)]

    # Calcola vettore medio PONDERATO in base alla frequenza nel training 
    freq_map = df_train[column + "_mapped"].value_counts().to_dict()
    emb_df["freq"] = emb_df["id"].map(freq_map)
    weighted_sum = (emb_df.drop(columns=["id", "freq"]).multiply(emb_df["freq"], axis=0)).sum()
    total_freq = emb_df["freq"].sum()
    default_vector = (weighted_sum / total_freq).values

    # Assegna gli embedding al training set 
    df_train = df_train.merge(emb_df.drop(columns=["freq"]).rename(columns={"id": column + "_mapped"}), 
                            on=column + "_mapped", how="left")

    # Embedding per validation set 
    df_val[column + "_mapped"] = df_val[column].map(id_map).astype(float).fillna(-1).astype(int)

    # Merge solo per valori noti
    df_val = df_val.merge(emb_df.drop(columns=["freq"]).rename(columns={"id": column + "_mapped"}), 
                        on=column + "_mapped", how="left")

    # Colonne embedding
    emb_cols = [f"{column}_emb_{i}" for i in range(emb_dim)]

    # Riempie NaN con il vettore medio ponderato
    for i, col in enumerate(emb_cols):
        df_val[col] = df_val[col].fillna(default_vector[i])

    # Stampa valori unseen nel validation 
    unseen = df_val[df_val[column + "_mapped"] == -1][column].unique()
    if len(unseen) > 0:
        print(f"[{column}] {len(unseen)} valori UNSEEN nel validation set: {unseen[:10]}...")

    # Pulisce colonne originali
    df_train = df_train.drop(columns=[column, column + "_mapped"])
    df_val = df_val.drop(columns=[column, column + "_mapped"])

    return df_train, df_val

class XGBonfolds:
    def __init__(self, df_full, path_dir_csv, params, embedding_data=None):
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
        self.embedding_data = embedding_data  # dizionario {col: (embedding_matrix, id_map)}

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
    


    def run(self, model_path_dir ,target_col='damage_grade', n_folds=5, save=False, trial_number=None):
        """
        Esegue il training e la valutazione di un modello XGBoost su k-fold cross-validation,
        salvando opzionalmente i modelli per ogni fold.

        Parametri:
        - model_path_dir (str): percorso in cui salvare i modelli allenati.
        - target_col (str): nome della colonna target nel dataset.
        - n_folds (int): numero di fold per la cross-validation.
        - save (bool): se True, salva i modelli in formato joblib nella directory specificata.
        - trial_number (int): per tenere traccia del trial di ottimizzazione che viene eseguito 

        Ritorna:
        - f1_scores (list of float): lista dei punteggi F1 per ciascun fold.
        - mean_f1 (float): media dei punteggi F1 su tutti i fold.
        """
        f1_scores = []

        for fold in range(1, n_folds + 1):
            print(f"\n===== Fold {fold} | Trial {trial_number} =====")

            train_idx = pd.read_csv(f"{self.path_dir_csv}/fold_{fold}_train.csv", header=None)[0].values
            val_idx = pd.read_csv(f"{self.path_dir_csv}/fold_{fold}_val.csv", header=None)[0].values

            df_train = self.df_full.iloc[train_idx].reset_index(drop=True)
            df_val = self.df_full.iloc[val_idx].reset_index(drop=True)

           

            # Applica embedding solo su geo_level_2_id e geo_level_3_id
            if self.embedding_data:
                for col, (emb_matrix, id_map) in self.embedding_data.items():
                    df_train, df_val = apply_fold_embedding(df_train, df_val, col, emb_matrix, id_map)

            

            X_train = df_train.drop(columns=[target_col])
            y_train = df_train[target_col] - 1  # classi 0,1,2
            X_val = df_val.drop(columns=[target_col])
            y_val = df_val[target_col] - 1
            
            



            


            model = XGBClassifier(**self.params)
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      verbose=False)

            if save:
                model_path = f"{model_path_dir}/xgb_model_version_3_fold_{fold}.joblib"
                dump(model, model_path)

            y_pred = model.predict(X_val)
            f1 = self.evaluate_f1_micro(y_val, y_pred)
            f1_scores.append(f1)

        mean_f1 = np.mean(f1_scores)
        print(f"F1-micro media su {n_folds} fold: {mean_f1:.4f}")
        return f1_scores, mean_f1
