import pandas as pd

features_path = 'C:/Users/emagi/Documents/richters_predictor/data/train_values.csv'    
labels_path   = 'C:/Users/emagi/Documents/richters_predictor/data/train_labels.csv'       

# Lettura dataset
df_features = pd.read_csv(features_path)
df_labels = pd.read_csv(labels_path)

# Unione di features e label in un unico dataset
df = pd.concat([df_features, df_labels['damage_grade']], axis=1)

print("Dimensione_dataset:", df.shape)

# Controllo missing values
missing = df.isnull().sum()
total_missing = missing.sum()

if total_missing > 0:
    print("Valori mancanti trovati:")
    for col in df.columns:
        if missing[col] > 0:
            print(f" - {col}: {missing[col]} valori NaN")

else:
    print("Nessun valore mancante trovato.")

print(df.dtypes)

# Eliminazione della colonna building_id 
df = df.drop(columns=['building_id'])

# Conversione dei tipi da object a category
object_cols = df.select_dtypes(include=['object']).columns.tolist()
df[object_cols] = df[object_cols].astype("category")

# Conversione tipi delle colonne geo_level da int64
geo_cols = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']
for col in geo_cols:
    if col in df.columns:
        df[col] = df[col].astype("category")

print(df.dtypes)

#  Stampa dei valori unici e frequenze per controllare la presenza di dati errati
print("Valori unici per colonna:")
for col in df.columns:
    print(f"{col}")
    val_counts = df[col].value_counts(dropna=False)  # gia mette mette i valori dalla freq più alta alla più bassa
    rel_freq = df[col].value_counts(normalize=True, dropna=False)
    n_unique = len(val_counts)
    print(f"{n_unique} valori unici")

    if n_unique <= 20:
        for val in val_counts.index:
            print(f"   {val} -> {val_counts[val]} occorrenze ({rel_freq[val]:.4f})")
    else:
        print("   Valori più frequenti:")
        for val in val_counts.head(10).index:
            print(f"    {val} -> {val_counts[val]} occorrenze ({rel_freq[val]:.4f})")

        print("   Valori meno frequenti:")
        for val in val_counts.tail(10).index:
            print(f"    {val} -> {val_counts[val]} occorrenze ({rel_freq[val]:.4f})")

    # outlier