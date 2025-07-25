# Richters_predictor

Sistema modulare completo per la **predizione del danno sismico agli edifici** (`damage_grade`) attraverso tecniche di **Machine Learning**, **embedding neurali**, **ottimizzazione iperparametrica con Optuna** e validazione K-Fold. Basato su CatBoost, LightGBM, Random Forest e XGBoost.

---

## Obiettivo

Predire la variabile `damage_grade` (classe 1, 2, 3) a partire da dati strutturali e geografici degli edifici, forniti da [IDRL Nepal Earthquake dataset](https://www.drivendata.org/competitions/57/nepal-earthquake/data/).

---

## Esecuzione degli script

Il progetto supporta **due modalità di esecuzione**:

### ESECUZIONE STANDALONE (consigliata)
E' possibile eseguire singolarmente i seguenti file `.py`:

- `data_cleaning.py`
- `geo_embedding.py`
- `opt_cat.py`
- `opt_lgmb.py`
- `opt_xgb.py`
- `submission_generator.py`

Questi file contengono un blocco `if __name__ == "__main__":` con un esempio di esecuzione diretta. Gli **import** sono scritti in modo da **permettere direttamente l'esecuzione del file in modalità standalone** in Visual Studio Code o terminale.

---

### ESECUZIONE COMPLETA VIA `main.py`

Il file `main.py` fornisce una **pipeline integrata** che unisce tutte le fasi del progetto, dalla pulizia dei dati alla generazione della submission finale.  
È pensato come esempio completo di flusso di lavoro.

> **Attenzione:** affinché `main.py` funzioni correttamente, bisogna modificare gli **import interni** dei file `.py` (sopra citati) come segue:

| Script da modificare       | Import originale                                                  | Import per `main.py`                                                        |
|----------------------------|-------------------------------------------------------------------|------------------------------------------------------------------------------|
| `opt_cat.py`               | `from data_cleaning import ...`<br>`from Catboost import ...`     | `from scripts.data_cleaning import ...`<br>`from scripts.Catboost import ...` |
| `opt_lgmb.py`              | `from data_cleaning import ...`<br>`from LightGBM import ...`     | `from scripts.data_cleaning import ...`<br>`from scripts.LightGBM import ...` |
| `opt_xgb.py`               | `from data_cleaning import ...`<br>`from XGBoost import ...`      | `from scripts.data_cleaning import ...`<br>`from scripts.XGBoost import ...`  |
| `submission_generator.py` | `from data_cleaning import ...`                                   | `from scripts.data_cleaning import ...`                                       |


Alternativamente, si possono duplicare questi file e creare una versione modificata per l’esecuzione di una specifica pipeline desiderata nel `main`.

---

## Flusso Operativo Completo (`main.py`)

1. **Data Preparation**:
   - Caricamento `train_values.csv` + `train_labels.csv`
   - Merge sul campo `damage_grade` e rimozione `building_id`

2. **Pulizia e Feature Engineering**:
   - Gestione missing values
   - Conversione tipi `object → category`
   - Identificazione e rimozione outlier multivariati via IQR
   - Salvataggio del dataset pulito (`clean_dataset.csv`)

3. **Cross-validation**:
   - Generazione indici `train/val` su 5 fold con `StratifiedKFold`
   - Salvataggio nella directory specificata di file .csv con gli indici

4. **Embedding Neurale (opzionale, per XGBoost)**:
   - Training di reti neurali per ciascuna `geo_level_*_id`
   - Salvataggio pesi (`.npy`) e mapping (`.pkl`)
   - Merge degli embedding nel dataset → `dataset_with_embedding.csv`

5. **Model Training con Optuna**:
   - Tuning automatico con `optuna.create_study()`
   - Training su 5 fold con i migliori iperparametri
   - Salvataggio modelli (`.joblib`) e `study` (`.pkl`)

6. **Generazione della Submission**:
   - Caricamento test set (`test_values.csv`)
   - Preprocessing (inclusa la gestione dell'embedding)
   - Predizione su tutti i 5 modelli → media delle probabilità
   - Output finale `submission_{model}.csv`

---

## Dettagli dei Moduli

### `data_cleaning.py`
- Conversione automatica tipi `object` → `category`
- Log dei valori unici e frequenze
- Rimozione outlier se presenti in oltre il 60% delle feature numeriche

---

### `cross_validation.py`
- `Skf` genera fold `train/val` con distribuzione stratificata delle classi
- Salva dei file CSV (`fold_i_train.csv` / `fold_i_val.csv`) contenenti gli indici dei campioni del training set e validation set per un i-esimo esperimento. 

---

### `Catboost.py`, `LightGBM.py`, `random_forest.py`, `XGBoost.py`
- Ogni file implementa una classe:
  - `CatBoostonfolds`
  - `LGBMonfolds`
  - `RFonfolds`
  - `XGBonfolds`
- Caricano gli indici di fold, eseguono il training, valutano con `F1-score (micro)`
- Opzione `save=True` per salvare i modelli

---

### `geo_embedding.py`
- Addestra una piccola rete neurale per ogni `geo_level_*_id`:
  - `Embedding → ReLU → Linear → output`
- Ottimizza con `CrossEntropyLoss` e `Adam`
- Supporta `early stopping` automatico
- Salva:
  - embedding matrix (`.npy`)
  - dizionario di mapping (`.pkl`)
- Funzione `apply_embedding()` che esegue il merge con il dataframe

---

### Optuna Hyperparameter Tuning
- `opt_cat.py`: tuning CatBoost
- `opt_lgmb.py`: tuning LightGBM
- `opt_xgb.py`: tuning XGBoost
- Funzioni:
  - `run_optuna(n_trials)` esegue tuning
  - `objective(trial)` valuta con `F1-micro` su 5 fold richiamando la rispettiva classe che addestra il modello
- Salva:
  - Studio (`.pkl`)
  - Modelli ottimizzati (`.joblib`)

---

### `submission_generator.py`
- Classe `Submitter`:
  - Preprocessing test set (con o senza embedding)
  - Caricamento modelli `.joblib` da 5 fold
  - Media delle probabilità (`predict_proba`)
  - `argmax` finale → `damage_grade`
  - Salva in `submission_{model}.csv`

---

## Modelli Supportati

| Modello       | Cross-Val | Optuna | Embedding | 
|---------------|-----------|--------|-----------|
| CatBoost      | ✅        | ✅    | ❌        |
| LightGBM      | ✅        | ✅    | ❌        | 
| XGBoost       | ✅        | ✅    | ✅        |
| Random Forest | ✅        | ❌    | ❌        |

---

## Setup Ambiente

### Requisiti Python

- Python ≥ 3.10 (consigliato 3.11)
- Compatibile con GPU (solo embedding se PyTorch installato con supporto CUDA)

### Installazione Dipendenze

```bash
pip install -r requirements.txt
