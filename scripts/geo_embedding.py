# generate_embeddings.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle

# === CONFIG ===
DATASET_PATH = 'C:/Users/emagi/Documents/richters_predictor/data/clean_dataset.csv'
OUTPUT_PATH = 'C:/Users/emagi/Documents/richters_predictor/data/dataset_with_embedding.csv'
EMBEDDING_DIR = 'C:/Users/emagi/Documents/richters_predictor/data/embeddings'
TARGET_COL = 'damage_grade'
HIDDEN_DIM = 32
EPOCHS = 20
PATIENCE = 2
BATCH_SIZE = 1024
LR = 0.01

# Numero di valori unici per ogni variabile categoriale (inseriti a mano)
UNIQUE_COUNTS = {
    'geo_level_1_id': 31,
    'geo_level_2_id': 1414,
    'geo_level_3_id': 11595
}


# Funzione per calcolare dimensione embedding a partire da valori unici
def compute_embedding_dim(n_unique):
    return min(50, round(n_unique ** 0.25))

class GeoEmbeddingDataset(Dataset):
    def __init__(self, ids, labels):
        self.ids = ids.astype(np.int64)
        self.labels = labels.astype(np.int64) - 1

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return torch.tensor(self.ids[idx]), torch.tensor(self.labels[idx])

class GeoEmbeddingNet(nn.Module):
    def __init__(self, num_ids, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_ids, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 3)

    def forward(self, geo_id):
        x = self.embedding(geo_id)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def train_embedding(df, column, num_unique):
    df[column + "_mapped"], uniques = pd.factorize(df[column])
    ids = df[column + "_mapped"].values
    labels = df[TARGET_COL].values

    dataset = GeoEmbeddingDataset(ids, labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    embedding_dim = compute_embedding_dim(num_unique)
    print(f" {column}: {num_unique} valori unici -> embedding_dim={embedding_dim}")

    num_ids = len(uniques)
    model = GeoEmbeddingNet(num_ids, embedding_dim, HIDDEN_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    patience_counter = 0

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for geo_id, label in dataloader:
            optimizer.zero_grad()
            output = model(geo_id)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f" {column} - Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping su {column} a epoch {epoch+1}")
                break

    embedding_matrix = model.embedding.weight.data.cpu().numpy()
    id_map = dict(zip(uniques, range(len(uniques))))

    # Salvataggio dei file
    np.save(f"{EMBEDDING_DIR}/embedding_{column}.npy", embedding_matrix)
    with open(f"{EMBEDDING_DIR}/id_map_{column}.pkl", "wb") as f:
        pickle.dump(id_map, f)

    return embedding_matrix, embedding_dim, id_map


def apply_embedding(df, column, embedding_matrix, emb_dim, id_map):
    df[column + "_mapped"] = df[column].map(id_map)
    emb_df = pd.DataFrame(embedding_matrix, columns=[f'{column}_emb_{i}' for i in range(emb_dim)])
    emb_df[column + "_mapped"] = emb_df.index
    df = df.merge(emb_df, on=column + "_mapped", how='left')
    df = df.drop(columns=[column, column + "_mapped"])
    return df


def main():
    df = pd.read_csv(DATASET_PATH)

    for column, n_unique in UNIQUE_COUNTS.items():
        print(f" Training embedding for {column}")
        emb_matrix, emb_dim, id_map = train_embedding(df, column, n_unique)
        df = apply_embedding(df, column, emb_matrix, emb_dim, id_map)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f" Dataset salvato con embedding in: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
