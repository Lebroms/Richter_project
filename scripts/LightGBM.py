import lightgbm as lgb
from sklearn.metrics import f1_score
from joblib import dump, load
import pandas as pd
import inspect
print(inspect.getfile(lgb.LGBMClassifier))


class LGBM:

    @staticmethod
    def train_and_save_model(df_train, df_val, target_col, model_path='lgbm_model.joblib', random_state=42):
        X_train = df_train.drop(columns=[target_col])
        y_train = df_train[target_col]
        X_val = df_val.drop(columns=[target_col])
        y_val = df_val[target_col]

        categorical_cols = X_train.select_dtypes(include='category').columns.tolist()

        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            class_weight='balanced',
            random_state=random_state,
            n_estimators=500,
            learning_rate=0.1,
            num_leaves=64,
            min_split_gain=0.01,
            min_child_samples=100,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            reg_alpha=1.0,  # lambda_l1
            reg_lambda=1.0,  # lambda_l2
            verbosity=-1
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='multi_logloss',
            callbacks=[lgb.early_stopping(stopping_rounds=40)]
        )
        dump(model, model_path)
        print(f"✅ Modello LGBM salvato in {model_path}")
        return model

    @staticmethod
    def load_and_predict(model_path, X_new):
        model = load(model_path)
        return model.predict(X_new)

    @staticmethod
    def evaluate_f1_micro(y_true, y_pred):
        f1_micro = f1_score(y_true, y_pred, average='micro')
        print(f"🎯 F1-micro: {f1_micro:.4f}")
        return f1_micro

    @staticmethod
    def run(df_full, path_dir='C:/Users/emagi/Documents/richters_predictor/data/cross_validation', target_col='damage_grade', n_folds=5):
        """
        Preprocessing + Addestramento + Valutazione per ogni fold.
        """
        f1_scores = []

        for fold in range(1, n_folds + 1):
            print(f"\n🔁 Fold {fold}")

            # 🔹 Caricamento indici
            train_idx = pd.read_csv(f"{path_dir}/fold_{fold}_train.csv", header=None)[0].values
            val_idx = pd.read_csv(f"{path_dir}/fold_{fold}_val.csv", header=None)[0].values

            # 🔹 Creazione dei DataFrame per training e validazione
            df_train = df_full.iloc[train_idx].reset_index(drop=True)
            df_val = df_full.iloc[val_idx].reset_index(drop=True)

            # 🔹 Colonne da eliminare (es. geo_level_1_id e geo_level_3_id)
            # drop_cols = ['geo_level_1_id', 'geo_level_3_id']
            #categorical_cols = df_train.select_dtypes(include='category').columns.tolist()

            # df_train = LGBM.preprocess_dataframe(df_train, drop_cols=drop_cols, categorical_cols=categorical_cols)
            # df_val = LGBM.preprocess_dataframe(df_val, drop_cols=drop_cols, categorical_cols=categorical_cols)

            # 🔹 Addestramento
            path_dir_model = "C:/Users/emagi/Documents/richters_predictor/models/"
            model_path = f"{path_dir_model}/lgbm_model_fold_{fold}.joblib"
            LGBM.train_and_save_model(df_train, df_val, target_col, model_path=model_path)

            # 🔹 Valutazione
            X_val = df_val.drop(columns=[target_col])
            y_val = df_val[target_col]
            y_pred = LGBM.load_and_predict(model_path, X_val)
            f1 = LGBM.evaluate_f1_micro(y_val, y_pred)
            f1_scores.append(f1)

            print(f"📁 Fold {fold} completato.")

        mean_f1 = sum(f1_scores) / len(f1_scores)
        print(f"\n📊 F1-micro media su {n_folds} fold: {mean_f1:.4f}")
        return f1_scores, mean_f1
