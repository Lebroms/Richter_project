import pandas as pd

class Data_cleaner():

    @staticmethod
    def missing_and_error_handler(df):
        """
        Stampa il totale dei valori mancanti nel dataset e il numero di valori mancanti per ogni colonna.
        Converte le colonne di tipo object in category.
        Converte il tipo delle colonne geo_level da int64 a category.
        Stampa dei valori unici e frequenze per controllare la presenza di dati errati

        Parameters:
            df (pd.DataFrame): Il Dataset da analizzare.

        Returns:
            df (pd.DataFrame): Il Datset pulito.
        """
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

        # Conversione dei tipi da object a category
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        df[object_cols] = df[object_cols].astype("category")

        # Conversione tipi delle colonne geo_level da int64 a category
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

        return df
    
    @staticmethod
    def remove_multivariate_outliers_iqr(df, threshold=0.6):
        """
        Rimuove record che sono outlier in più del 60% delle feature numeriche (escludendo flag binari).
        
        Parametri:
            df (pd.DataFrame): Il dataset originale
        
        Ritorna:
            df_clean (pd.DataFrame): Il dataset pulito
        """

        # Trova colonne numeriche (non flag binari e label)
        numeric_cols = []
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                unique_vals = df[col].nunique()
                if unique_vals > 3:  # 3 così escludo anche label
                    numeric_cols.append(col)

        print("Colonne numeriche su cui applico IQR:")
        print(numeric_cols)

        # Crea un DataFrame con gli stessi indici per memorizzare gli "outlier flag"
        outlier_flags = pd.DataFrame(index=df.index)

        # Calcola se ogni valore è un outlier secondo l'IQR per ogni colonna
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)         # 25° percentile
            Q3 = df[col].quantile(0.75)         # 75° percentile
            IQR = Q3 - Q1                       # Intervallo interquartile
            lower_bound = Q1 - 1.5 * IQR        # limite inferiore
            upper_bound = Q3 + 1.5 * IQR        # limite superiore

            # True se il valore è fuori dai limiti, outlier_flag è un df con flag per ogni valore di ogni campione
            outlier_flags[col] = (df[col] < lower_bound) | (df[col] > upper_bound)

        # Conta quante flags true ci sono per ogni riga
        outlier_counts = outlier_flags.sum(axis=1)

        # Soglia per considerare se il campione è outlier globale, cioè considerando la maggior parte delle features
        soglia = len(numeric_cols)*threshold
        outliers = outlier_counts > soglia

        print(f"Outlier riconosciuti che verranno eliminati dal dataset: {outliers.sum()} su {len(df)}")

        # Rimuovi le righe outliers
        df_clean = df[outliers == False]
        return df_clean

if __name__ == "__main__":
    
    features_path = 'C:/Users/emagi/Documents/richters_predictor/data/train_values.csv'
    labels_path = 'C:/Users/emagi/Documents/richters_predictor/data/train_labels.csv'
    output_path = 'C:/Users/emagi/Documents/richters_predictor/data/clean_dataset.csv'

    # Caricamento e unione dataset
    df_features = pd.read_csv(features_path)
    df_labels = pd.read_csv(labels_path)
    df = pd.concat([df_features, df_labels['damage_grade']], axis=1)
    df = df.drop(columns=['building_id'])

    print("Dataset caricato. Dimensioni iniziali:", df.shape)

    # Cleaning
    df = Data_cleaner.missing_and_error_handler(df)
    df = Data_cleaner.remove_multivariate_outliers_iqr(df)

'''    # Salvataggio
    df.to_csv(output_path, index=False)
    print(f"Dataset pulito salvato in: {output_path}")'''

