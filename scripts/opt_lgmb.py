import optuna
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score
from joblib import dump
from data_cleaning import Data_cleaner

# === CONFIG ===
TARGET_COL = "damage_grade"
N_FOLDS = 5
DATA_PATH = "C:/Users/emagi/Documents/richters_predictor/data/cross_validation"
STUDY_PATH = "C:/Users/emagi/Documents/richters_predictor/models/optuna_study_lgbm.pkl"

# === Funzione principale ===
def run_optuna(df_full, n_trials=40):

    def objective(trial):
        f1_scores = []

        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'verbosity': -1,
            'n_estimators': 1500,
            'random_state' : 42,
            'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int("num_leaves", 31, 255),
            'max_depth': trial.suggest_int("max_depth", 4, 15),
            'min_child_samples': trial.suggest_int("min_child_samples", 20, 300),
            'feature_fraction': trial.suggest_float("feature_fraction", 0.6, 1.0),
            'bagging_fraction': trial.suggest_float("bagging_fraction", 0.7, 1.0),
            'bagging_freq': trial.suggest_int("bagging_freq", 1, 5),
            'lambda_l1': trial.suggest_float("lambda_l1", 0.0, 5.0),
            'lambda_l2': trial.suggest_float("lambda_l2", 0.0, 5.0)
        }


        for fold in range(1, N_FOLDS + 1):
            train_idx = pd.read_csv(f"{DATA_PATH}/fold_{fold}_train.csv", header=None)[0].values
            val_idx = pd.read_csv(f"{DATA_PATH}/fold_{fold}_val.csv", header=None)[0].values

            df_train = df_full.iloc[train_idx].reset_index(drop=True)
            df_val = df_full.iloc[val_idx].reset_index(drop=True)

            X_train = df_train.drop(columns=[TARGET_COL])
            y_train = df_train[TARGET_COL]
            X_val = df_val.drop(columns=[TARGET_COL])
            y_val = df_val[TARGET_COL]

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=40)]
            )

            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='micro')
            f1_scores.append(f1)

        return np.mean(f1_scores)

    # === Ottimizzazione ===
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    study.optimize(objective, n_trials=n_trials, n_jobs=6)

    print("Best F1-micro:", study.best_value)
    print("Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # === Salva lo study ===
    dump(study, STUDY_PATH)
    print(f"Study salvato in: {STUDY_PATH}")

    return study


def train_and_save_best_model(df_full, best_params):
    """
    Allena 5 modelli LGBM sui fold specifici usati per Optuna, con early stopping
    valutato sui rispettivi validation set. Salva ogni modello separatamente.
    """
    for fold in range(1, N_FOLDS + 1):
        print(f"Training modello fold {fold}...")

        # Carica gli indici
        train_idx = pd.read_csv(f"{DATA_PATH}/fold_{fold}_train.csv", header=None)[0].values
        val_idx = pd.read_csv(f"{DATA_PATH}/fold_{fold}_val.csv", header=None)[0].values

        # Crea subset
        df_train = df_full.iloc[train_idx].reset_index(drop=True)
        df_val = df_full.iloc[val_idx].reset_index(drop=True)

        X_train = df_train.drop(columns=[TARGET_COL])
        y_train = df_train[TARGET_COL]
        X_val = df_val.drop(columns=[TARGET_COL])
        y_val = df_val[TARGET_COL]

        # Crea modello
        model = lgb.LGBMClassifier(
            **best_params,
            objective='multiclass',
            num_class=3,
            metric='multi_logloss',
            n_estimators=1500,
            verbosity=-1,
            random_state=42
        )

        # Addestra con early stopping sul validation dell'esoerimento
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=40)]
        )

        # Salva il modello
        model_path = f"C:/Users/emagi/Documents/richters_predictor/models/lgbm_model_fold_{fold}.joblib"
        dump(model, model_path)
        print(f"Modello 1 {fold} salvato in: {model_path}")

    print("Tutti i modelli LightGBM con i best_params sono stati addestrati e salvati.")



if __name__ == "__main__":
    dataset_path = 'C:/Users/emagi/Documents/richters_predictor/data/clean_dataset.csv'
    df = pd.read_csv(dataset_path)
    df = Data_cleaner.missing_and_error_handler(df)
    print(df.dtypes)
    print(f"Dataset caricato: {df.shape}")
    study = run_optuna(df, n_trials=40)
    train_and_save_best_model(df, study.best_params)



