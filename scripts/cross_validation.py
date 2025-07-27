import pandas as pd
from sklearn.model_selection import StratifiedKFold


class Skf():
     

    def __init__(self, df_full, path_dir_csv,target_col):
        
        self.df_full = df_full
        self.path_dir_csv = path_dir_csv
        self.target_col=target_col

    def get_train_val_index_folds(self, n_splits=5, random_state=42):
        """
        Genera i fold stratificati e salva solo gli indici di train e validation
        come file CSV (una colonna, senza header).

        Parametri:
            n_splits (int): numero di fold
            random_state (int): seed per riproducibilità

        Ritorna:
            List[Tuple[np.ndarray, np.ndarray]]: lista di (train_idx, val_idx)
        """
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        y = self.df_full[self.target_col]
        folds = []

        for i, (train_idx, val_idx) in enumerate(skf.split(self.df_full, y)):
            train_path = self.path_dir_csv + f"fold_{i+1}_train.csv"
            val_path = self.path_dir_csv + f"fold_{i+1}_val.csv"

            # Salva solo gli indici, senza header e senza index
            pd.DataFrame(train_idx).to_csv(train_path, index=False, header=False)
            pd.DataFrame(val_idx).to_csv(val_path, index=False, header=False)

            print(f"Salvati: {train_path}, {val_path}")
            folds.append((train_idx, val_idx))
        
        return folds

        

    







