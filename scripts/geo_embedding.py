import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle

# CONFIG
DATASET_PATH = 'C:/Users/emagi/Documents/richters_predictor/data/clean_dataset.csv'
OUTPUT_PATH = 'C:/Users/emagi/Documents/richters_predictor/data/dataset_with_emb.csv'
EMBEDDING_DIR = 'C:/Users/emagi/Documents/richters_predictor/data/embeddings'
TARGET_COL = 'damage_grade'
HIDDEN_DIM = 32
EPOCHS = 20
PATIENCE = 2
BATCH_SIZE = 1024
LR = 0.01

# Numero di valori unici per ogni variabile categoriale
UNIQUE_COUNTS = {
    'geo_level_1_id': 31,
    'geo_level_2_id': 1414,
    'geo_level_3_id': 11595
}

# Funzione per calcolare dimensione embedding a partire da valori unici
def compute_embedding_dim(n_unique):
    """
    Calcola la dimensione ottimale dell'embedding dato il numero di valori unici.

    Parametri:
    - n_unique (int): numero di valori unici della variabile categoriale.

    Ritorna:
    - int: dimensione dell'embedding (massimo 50).
    """
    return min(50, round(n_unique ** 0.25))

class GeoEmbeddingDataset(Dataset):
    """
    Dataset personalizzato per PyTorch che associa ID geografici e relative etichette target.
    """
    def __init__(self, ids, labels):
        """
        Costruttore

        Parametri:
        - ids (np.ndarray): array di interi che rappresentano gli ID mappati.
        - labels (np.ndarray): array di etichette di classe (interi).

        Ritorna:
        - None
        """
        self.ids = ids.astype(np.int64)
        self.labels = labels.astype(np.int64) - 1

    def __len__(self):
        """
        Restituisce la lunghezza del dataset (numero di campioni).

        Parametri:
        - None

        Ritorna:
        - int: numero di campioni nel dataset
        """
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Restituisce un singolo campione dato l'indice.

        Parametri:
        - idx (int): indice del campione da restituire.

        Ritorna:
        - tuple(torch.Tensor, torch.Tensor): (ID categoriale, etichetta target)
        """
        return torch.tensor(self.ids[idx]), torch.tensor(self.labels[idx])

class GeoEmbeddingNet(nn.Module):
    """
    Rete semplice con uno strato di embedding e due layer fully connected.
    """
    def __init__(self, num_ids, embedding_dim, hidden_dim):
        """
        Costruttore

        Parametri:
        - num_ids (int): numero di ID unici nella colonna categoriale.
        - embedding_dim (int): dimensione del vettore di embedding.
        - hidden_dim (int): numero di neuroni nello strato nascosto.

        Ritorna:
        - None
        """
        super().__init__()
        self.embedding = nn.Embedding(num_ids, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 3)

    def forward(self, geo_id):
        """
        Esegue il forward pass del modello.

        Parametri:
        - geo_id (torch.Tensor): batch di ID categoriali.

        Ritorna:
        - torch.Tensor: logits (valori grezzi senza softmax) di output per ciascuna delle 3 classi.
        """
        x = self.embedding(geo_id)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def train_embedding(df, column, num_unique):
    """
    Addestra un embedding per una variabile categoriale tramite classificazione supervisata.

    Parametri:
    - df (pd.DataFrame): dataframe contenente i dati e la colonna da embeddare.
    - column (str): nome della colonna categoriale.
    - num_unique (int): numero di valori unici nella colonna.

    Ritorna:
    - embedding_matrix (np.ndarray): matrice degli embedding appresi.
    - emb_dim (int): dimensione degli embedding.
    - id_map (dict): mappa valore originale → ID numerico.
    """
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
    """
    Applica l'embedding appreso al dataset, sostituendo la colonna originale con colonne numeriche.

    Parametri:
    - df (pd.DataFrame): dataframe originale.
    - column (str): nome della colonna su cui applicare l'embedding.
    - embedding_matrix (np.ndarray): matrice degli embedding appresi.
    - emb_dim (int): dimensione dell'embedding.
    - id_map (dict): mappa valore → ID numerico.

    Ritorna:
    - pd.DataFrame: dataframe con colonne di embedding (lo salva anche in OUTPUT_PATH).
    """
    df[column + "_mapped"] = df[column].map(id_map)
    emb_df = pd.DataFrame(embedding_matrix, columns=[f'{column}_emb_{i}' for i in range(emb_dim)])
    emb_df[column + "_mapped"] = emb_df.index
    df = df.merge(emb_df, on=column + "_mapped", how='left')
    df = df.drop(columns=[column, column + "_mapped"])

    # salvataggio del dataset con emb
    df.to_csv(OUTPUT_PATH, index=False)
    print(f" Dataset salvato con embedding in: {OUTPUT_PATH}")
    return df


def main():
    df = pd.read_csv(DATASET_PATH)

    for column, n_unique in UNIQUE_COUNTS.items():
        print(f" Training embedding per {column}")
        emb_matrix, emb_dim, id_map = train_embedding(df, column, n_unique)
        df = apply_embedding(df, column, emb_matrix, emb_dim, id_map)

if __name__ == '__main__':
    main()
