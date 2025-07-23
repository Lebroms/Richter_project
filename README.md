
Richter’s Predictor
=======
# richters_predictor

Sistema modulare completo per la **predizione del danno sismico agli edifici** (`damage_grade`) attraverso tecniche avanzate di **Machine Learning**, **embedding neurali**, **ottimizzazione iperparametrica con Optuna** e validazione K-Fold. Basato su CatBoost, LightGBM, Random Forest e XGBoost.

---

## Obiettivo

Predire la variabile `damage_grade` (classe 1, 2, 3) a partire da dati strutturali e geografici degli edifici, forniti da [IDRL Nepal Earthquake dataset](https://www.drivendata.org/competitions/57/nepal-earthquake/data/).



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
   - Salvataggio nella directory specificata di file .csv degli indici del dataset estratti dalla       stratified k cross validation

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
- Funziona sia standalone che integrato in `main.py`

---

### `cross_validation.py`
- `Skf` genera fold `train/val` con distribuzione stratificata delle classi
- Salva 10 CSV (`fold_i_train.csv` / `fold_i_val.csv`) per i moduli di training

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
- Funzione `apply_embedding()` esegue il merge con il dataframe

---

### Optuna Hyperparameter Tuning
- `opt_cat.py`: tuning CatBoost
- `opt_lgmb.py`: tuning LightGBM
- `opt_xgb.py`: tuning XGBoost
- Funzioni:
  - `run_optuna(n_trials)` esegue tuning
  - `objective(trial)` valuta con `F1-micro` su 5 fold
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
| CatBoost      | ✅        | ✅     | ❌        |
| LightGBM      | ✅        | ✅     | ❌        | 
| XGBoost       | ✅        | ✅     | ✅        |
| Random Forest | ✅        | ❌     | ❌        |

---

## Setup Ambiente

### Requisiti Python

- Python ≥ 3.10 (consigliato 3.11)
- Compatibile con GPU (solo embedding se PyTorch installato con supporto CUDA)

### Installazione Dipendenze

```bash
pip install -r requirements.txt


