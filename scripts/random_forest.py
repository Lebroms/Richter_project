from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import f1_score

class RF:

    @staticmethod
    def random_forest_with_folds(df, target_col, folds, random_state=42):
        """
        Esegue Random Forest su fold specificati, con Ordinal Encoding
        e iperparametri personalizzati. Usa F1 micro come metrica.

        Parametri:
            df (pd.DataFrame): Dataset completo
            target_col (str): Nome colonna target
            folds (List[Tuple]): Lista di (train_idx, test_idx)
            random_state (int): Seed per RandomForest

        Ritorna:
            List[float]: F1-micro per ogni fold
        """
        df = df.copy()
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Encoding delle categoriche
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        encoder = OrdinalEncoder()
        X[cat_cols] = encoder.fit_transform(X[cat_cols])

        f1_scores = []

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Modello Random Forest configurato
            model = RandomForestClassifier(
                n_estimators=100,
                criterion='gini',
                min_impurity_decrease=0.005,
                max_features='log2',
                class_weight='balanced',
                random_state=random_state
            )

            model.fit(X_train, y_train)

            print(f"[Fold {fold_idx + 1}] Alberi costruiti: {len(model.estimators_)}")

            y_pred = model.predict(X_test)
            f1_micro = f1_score(y_test, y_pred, average='micro')
            f1_scores.append(f1_micro)

        print("F1-micro per fold:", f1_scores)
        print("F1-micro media:", sum(f1_scores)/len(f1_scores))

        return f1_scores
