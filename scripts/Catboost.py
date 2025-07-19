from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from joblib import dump, load
import pandas as pd
import numpy as np

class CatBoostonfolds:

    def __init__(self, df_full):
        self.df_full = df_full

        # Parametri fissi del modello
        self.model_dir = "C:/Users/lscor/OneDrive/Magistrale/F_Intelligenza_artificiale/Richter_project/models/"
        self.path_dir_csv = "C:/Users/lscor/OneDrive/Magistrale/F_Intelligenza_artificiale/Richter_project/dati/cross_validation/"

        self.params = {
            'iterations': 1000,
            'learning_rate': 0.1,
            'min_data_in_leaf': 20,              
            'loss_function': 'MultiClass',
            'eval_metric': 'TotalF1:average=Micro',  
            'random_seed': 42,
            'verbose': 0,
            'early_stopping_rounds': 30
        }


    @staticmethod
    def evaluate_f1_micro(y_true, y_pred):
        f1 = f1_score(y_true, y_pred, average='micro')
        print(f"🎯 F1-micro: {f1:.4f}")
        return f1

    def preprocess(self, df):
        
        cat_cols = ['geo_level_3_id']
        other_cats = df.select_dtypes(include=['object', 'category']).columns.tolist()
        cat_cols += [col for col in other_cats if col not in cat_cols]

        df[cat_cols] = df[cat_cols].astype(str)
        df = df.drop(columns=['geo_level_1_id','geo_level_2_id'])
        return df, cat_cols

    def run(self, target_col='damage_grade', n_folds=5):
        f1_scores = []

        for fold in range(1, n_folds + 1):
            print(f"\n🔁 Fold {fold}")

            # Leggi indici
            train_idx = pd.read_csv(f"{self.path_dir_csv}/fold_{fold}_train.csv", header=None)[0].values
            val_idx = pd.read_csv(f"{self.path_dir_csv}/fold_{fold}_val.csv", header=None)[0].values

            # Estrai subset
            df_train = self.df_full.iloc[train_idx].reset_index(drop=True)
            df_val = self.df_full.iloc[val_idx].reset_index(drop=True)

            # Preprocessing e cast
            df_train, cat_cols = self.preprocess(df_train)
            df_val, _ = self.preprocess(df_val)

            X_train = df_train.drop(columns=[target_col])
            y_train = df_train[target_col]
            X_val = df_val.drop(columns=[target_col])
            y_val = df_val[target_col]

            model = CatBoostClassifier(**self.params)
            model.fit(X_train, y_train,
                      eval_set=(X_val, y_val),
                      cat_features=cat_cols)

            model_path = f"{self.model_dir}/catboost_model_fold_{fold}.joblib"
            dump(model, model_path)
            print(f"✅ Salvato modello in {model_path}")

            y_pred = model.predict(X_val)
            f1 = self.evaluate_f1_micro(y_val, y_pred)
            f1_scores.append(f1)

        mean_f1 = np.mean(f1_scores)
        print(f"\n📊 F1-micro media su {n_folds} fold: {mean_f1:.4f}")
        return f1_scores, mean_f1
