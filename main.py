import pandas as pd

from scripts.data_cleaning import Data_cleaner
from scripts.cross_validation import Skf

from scripts.opt_cat import Catboost_tuning

from scripts.submission_generator import Submitter


#path dove ci sono i vari training e validation set con gli indici estratti dalla skf
path_dir_csv='C:/Users/emagi/Documents/richters_predictor/data/cross_validation/'

# Path di salvataggio del modello finale
model_path = "C:/Users/emagi/Documents/richters_predictor/models/"

# Path di salvataggio dello study Optuna
study_path = "C:/Users/emagi/Documents/richters_predictor/models/optuna_study_catboost.pkl"  # Path di salvataggio dello study Optuna

model_type = "xgboost"  # "lightgbm", "catboost" 

#path di salvataggio del file contenente il test set (senza label)
test_value_path = 'C:/Users/emagi/Documents/richters_predictor/data/test_values.csv'

#path di salvataggio del file di submission preso dal sito della challenge da modificare
submission_format_path = 'C:/Users/emagi/Documents/richters_predictor/data/submission_format.csv'

#path di salvataggio del file di submission da caricare 
submission_output_path = f"C:/Users/emagi/Documents/richters_predictor/data/submission_{model_type}.csv"

#path di salvataggio del dataset con embedding (nel caso di xgboost)
embedding_dir = "C:/Users/emagi/Documents/richters_predictor/data/embeddings"


#path dei file di train
features_path = 'C:/Users/emagi/Documents/richters_predictor/data/train_values.csv'  
labels_path= 'C:/Users/emagi/Documents/richters_predictor/data/train_labels.csv'

#path di dove salvare il dataset pulito
output_path = 'C:/Users/emagi/Documents/richters_predictor/data/clean_dataset.csv'

target_col='damage_grade'

df_features = pd.read_csv(features_path)
df_labels = pd.read_csv(labels_path)
df = pd.concat([df_features, df_labels['damage_grade']], axis=1)
df = df.drop(columns=['building_id'])


print("Dataset caricato. Dimensioni iniziali:", df.shape)


# Cleaning
df = Data_cleaner.missing_and_error_handler(df)
df = Data_cleaner.remove_multivariate_outliers_iqr(df)


# Salvataggio
df.to_csv(output_path, index=False)
print(f"Dataset pulito salvato in: {output_path}")



#Stratified K-Cross Validation

cross_validator=Skf(df,path_dir_csv,target_col)
n_splits=5
cross_validator.get_train_val_index_folds(n_splits)

print('I file per eseguire la cross validation sono stati caricati correttamente')




#Esempio di implementazione con Catboost da inserire con il modello che si vuole eseguire

model_tuner= Catboost_tuning(df,target_col,n_splits,path_dir_csv,model_path,study_path)
model_tuner.run_optuna()



model_path = {
            "catboost": [f"C:/Users/emagi/Documents/richters_predictor/models/catboost_model_fold_{i}.joblib" for i in range(1, 6)],
            "lightgbm": [f"C:/Users/emagi/Documents/richters_predictor/models/lightgbm_model_fold_{i}.joblib" for i in range(1, 6)],
            "xgboost": [f"C:/Users/emagi/Documents/richters_predictor/models/xgb_model_fold_{i}.joblib" for i in range(1, 6)],
        }
    
sub = Submitter (model_type, test_value_path, submission_format_path, submission_output_path, embedding_dir)
sub.generate_submission(model_path)