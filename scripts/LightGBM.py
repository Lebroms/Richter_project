import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from numpy import mean, std


train_df = pd.read_csv('train_values.csv')
test_df = pd.read_csv('test_values.csv')
labels_df = pd.read_csv('train_labels.csv')


categorical_features = [
    'land_surface_condition',
    'foundation_type',
    'roof_type',
    'ground_floor_type',
    'other_floor_type',
    'position',
    'plan_configuration',
    'legal_ownership_status',
    'geo_level_1_id',
    'geo_level_2_id',
    'geo_level_3_id'
]


for col in categorical_features:
    train_df[col] = train_df[col].astype('category')
    test_df[col] = test_df[col].astype('category')


X = train_df.copy()
y = labels_df['damage_grade'].copy()

n_splits = 5  

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

f1_micro_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n===== Fold {fold + 1} =====")
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        min_child_samples=20,
        reg_lambda=3.0,
        subsample=0.8,
        random_state=42
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='multi_logloss',
        categorical_feature=categorical_features,
    )

    y_val_pred = model.predict(X_val)
    f1_micro = f1_score(y_val, y_val_pred, average='micro')
    print(f"F1 micro (Fold {fold + 1}): {f1_micro:.4f}")

    f1_micro_scores.append(f1_micro)

print(f"\nF1 micro average: {mean(f1_micro_scores):.4f}")
print(f"F1 micro std: {std(f1_micro_scores):.4f}")
