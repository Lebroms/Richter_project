import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from joblib import dump, load
import pandas as pd
import numpy as np

class XGBWithEmbedding:
    def __init__(self, df_full):
        # Dataset completo già fornito in input
        self.df_full = df_full

        # Parametri fissi
        self.embedding_dim = 10
        self.hidden_dim = 32
        self.epochs = 5
        self.batch_size = 1024
        self.lr = 0.01
        self.model = None

        # Path fissi
        self.path_dir_csv = "C:/Users/lscor/OneDrive/Magistrale/F_Intelligenza_artificiale/Richter_project/dati/cross_validation/"
        self.model_dir = "C:/Users/lscor/OneDrive/Magistrale/F_Intelligenza_artificiale/Richter_project/models/"

    class GeoEmbeddingDataset(Dataset):
        def __init__(self, df):
            self.geo_ids = df['geo_level_3_id'].values.astype(np.int64)
            self.labels = df['damage_grade'].values.astype(np.int64) - 1

        def __len__(self):
            return len(self.geo_ids)

        def __getitem__(self, idx):
            return torch.tensor(self.geo_ids[idx]), torch.tensor(self.labels[idx])

    class GeoEmbeddingNet(nn.Module):
        def __init__(self, num_geo_ids, embedding_dim, hidden_dim):
            super().__init__()
            self.embedding = nn.Embedding(num_geo_ids, embedding_dim)
            self.fc1 = nn.Linear(embedding_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, 3)

        def forward(self, geo_id):
            x = self.embedding(geo_id)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    def fit_embedding(self, df):
        dataset = self.GeoEmbeddingDataset(df)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        num_geo_ids = df['geo_level_3_id'].max() + 1
        self.model = self.GeoEmbeddingNet(num_geo_ids, self.embedding_dim, self.hidden_dim)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for geo_id, label in dataloader:
                optimizer.zero_grad()
                output = self.model(geo_id)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {total_loss:.4f}")

    def transform_embedding(self, df):
        embedding_matrix = self.model.embedding.weight.data.cpu().numpy()
        emb_dim = embedding_matrix.shape[1]
        embedding_df = pd.DataFrame(
            embedding_matrix,
            columns=[f'geo3_emb_{i}' for i in range(emb_dim)]
        )
        embedding_df['geo_level_3_id'] = embedding_df.index
        df_merged = df.merge(embedding_df, on='geo_level_3_id', how='left')
        return df_merged

    def fit_transform_embedding(self, df):
        self.fit_embedding(df)
        return self.transform_embedding(df)

    @staticmethod
    def train_xgb(df, target_col, model_path='xgb_model.joblib', random_state=42):
        X = df.drop(columns=[target_col])
        y = df[target_col] - df[target_col].min()

        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
        X[cat_cols] = X[cat_cols].astype("category")

        model = XGBClassifier(
            enable_categorical=True,
            n_estimators=100,
            learning_rate=0.1,
            gamma=1.3,
            subsample=0.8,
            colsample_bytree=0.5,
            reg_alpha=0.01,
            reg_lambda=0.06,
            use_label_encoder=False,
            objective='multi:softmax',
            eval_metric='auc',
            random_state=random_state
        )

        model.fit(X, y)
        dump(model, model_path)
        print(f"✅ Modello XGBoost salvato in {model_path}")
        return model

    @staticmethod
    def predict_xgb(model_path, X_new):
        model = load(model_path)
        cat_cols = X_new.select_dtypes(include=["object"]).columns.tolist()
        X_new[cat_cols] = X_new[cat_cols].astype("category")
        return model.predict(X_new)

    @staticmethod
    def evaluate_f1(y_true, y_pred):
        f1_micro = f1_score(y_true, y_pred, average='micro')
        print(f"🎯 F1-micro: {f1_micro:.4f}")
        return f1_micro

    def run(self):
        df_embedded = self.fit_transform_embedding(self.df_full)

        # Rimuovi geo-level ID dopo aver fatto embedding
        df_embedded = df_embedded.drop(columns=[
            'geo_level_1_id',
            'geo_level_2_id',
            'geo_level_3_id'
        ])

        f1_scores = []
        for fold in range(1, 6):
            print(f"\n🔁 Fold {fold}")
            train_idx = pd.read_csv(f"{self.path_dir_csv}/fold_{fold}_train.csv", header=None)[0].values
            val_idx = pd.read_csv(f"{self.path_dir_csv}/fold_{fold}_val.csv", header=None)[0].values

            df_train = df_embedded.iloc[train_idx].reset_index(drop=True)
            df_val = df_embedded.iloc[val_idx].reset_index(drop=True)

            df_train, df_val = df_train.align(df_val, join='left', axis=1, fill_value=0)

            model_path = f"{self.model_dir}/xgb_emb_model_fold_{fold}.joblib"
            self.train_xgb(df_train, 'damage_grade', model_path=model_path)

            X_val = df_val.drop(columns=['damage_grade'])
            y_val = df_val['damage_grade']
            y_pred = self.predict_xgb(model_path, X_val)
            y_pred += 1

            f1 = self.evaluate_f1(y_val, y_pred)
            f1_scores.append(f1)

        mean_f1 = sum(f1_scores) / len(f1_scores)
        print(f"\n📊 F1-micro media su 5 fold: {mean_f1:.4f}")
        return f1_scores, mean_f1
