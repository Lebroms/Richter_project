from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from joblib import dump, load
from kmodes.kprototypes import KPrototypes
import pandas as pd

class RF:

    @staticmethod
    def train_and_save_model(df, target_col, model_path='rf_model.joblib', random_state=42):
        X = df.drop(columns=[target_col])
        y = df[target_col]

        model = RandomForestClassifier(
            n_estimators=100,
            max_features='log2',
            class_weight='balanced',
            random_state=random_state
        )

        model.fit(X, y)
        dump(model, model_path)
        print(f"✅ Modello salvato in {model_path}")
        return model

    @staticmethod
    def load_and_predict(model_path, X_new):
        model = load(model_path)
        return model.predict(X_new)

    @staticmethod
    def evaluate_f1_micro(y_true, y_pred):
        """
        Calcola e stampa la F1-micro tra y_true e y_pred.
        """
        f1_micro = f1_score(y_true, y_pred, average='micro')
        print(f"🎯 F1-micro: {f1_micro:.4f}")
        return f1_micro

    @staticmethod
    def preprocess_and_train_on_folds(path_dir, target_col='damage_grade', n_folds=5):
        
        

        f1_scores = []

        for fold in range(1, n_folds + 1):
            print(f"\n🔁 Fold {fold}")

            # Carica i dati
            train_file = f"{path_dir}/fold_{fold}_train.csv"
            val_file = f"{path_dir}/fold_{fold}_val.csv"

            df_train = pd.read_csv(train_file)
            df_val = pd.read_csv(val_file)

            df_train = df_train.drop(columns=['geo_level_1_id','geo_level_3_id'])
            df_val = df_val.drop(columns=['geo_level_1_id','geo_level_3_id'])

            df_train['geo_level_2_id']=df_train['geo_level_2_id'].astype("category")
            df_val['geo_level_2_id']=df_val['geo_level_2_id'].astype("category")


        

            df_train = pd.get_dummies(df_train, columns=df_train.select_dtypes(include=['object', 'category']).columns)
            df_val = pd.get_dummies(df_val, columns=df_val.select_dtypes(include=['object', 'category']).columns)



            # Allineamento delle colonne (potrebbero esserci categorie mancanti in val)
            df_train, df_val = df_train.align(df_val, join='left', axis=1, fill_value=0)


            path_dir_model='C:/Users/lscor/OneDrive/Magistrale/F_Intelligenza_artificiale/Richter_project/models/'
            # Allenamento modello
            model_name = f"{path_dir_model}/rf_model_fold_{fold}.joblib"
            RF.train_and_save_model(df_train, target_col, model_path=model_name)

            # Predizione sul validation set
            X_val = df_val.drop(columns=[target_col])
            y_val = df_val[target_col]
            y_pred = RF.load_and_predict(model_name, X_val)

            # Valutazione F1 score
            f1 = RF.evaluate_f1_micro(y_val, y_pred)
            f1_scores.append(f1)

            print(f"📁 Fold {fold} completato.")

        # Media F1 finale
        mean_f1 = sum(f1_scores) / len(f1_scores)
        print(f"\n📊 F1-micro media su {n_folds} fold: {mean_f1:.4f}")
        return f1_scores, mean_f1
