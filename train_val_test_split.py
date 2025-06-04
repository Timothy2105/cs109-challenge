# train_val_test_split.py

import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # Load the combined feature CSV
    df = pd.read_csv('all_features.csv')
    
    # Separate features (X) from labels (y)
    X = df.drop(columns=['filename', 'composer'])
    y = df['composer']

    # First split: 70% train / 30% temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    # Second split: split temp 50/50 → 15% val, 15% test overall
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    # Re‐attach filename and composer columns so we can inspect them if needed
    train_idx = X_train.index
    val_idx   = X_val.index
    test_idx  = X_test.index

    df_train = df.loc[train_idx].reset_index(drop=True)
    df_val   = df.loc[val_idx].reset_index(drop=True)
    df_test  = df.loc[test_idx].reset_index(drop=True)

    # Save to CSV
    df_train.to_csv('train_set.csv', index=False)
    df_val.to_csv('val_set.csv', index=False)
    df_test.to_csv('test_set.csv', index=False)

    print(
        f"Train: {len(df_train)} examples, "
        f"Val: {len(df_val)} examples, "
        f"Test: {len(df_test)} examples"
    )

if __name__ == '__main__':
    main()
