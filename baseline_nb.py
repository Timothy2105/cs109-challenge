# baseline_nb.py

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def main():
    # Load train/val/test
    df_train = pd.read_csv('train_set.csv')
    df_val   = pd.read_csv('val_set.csv')
    df_test  = pd.read_csv('test_set.csv')

    # Use only the 12 pitch-class columns (pc_0 … pc_11)
    pc_cols = [f'pc_{i}' for i in range(12)]
    X_train = df_train[pc_cols]
    y_train = df_train['composer']
    X_val   = df_val[pc_cols]
    y_val   = df_val['composer']
    X_test  = df_test[pc_cols]
    y_test  = df_test['composer']

    # Train MultinomialNB
    nb = MultinomialNB(alpha=1.0)
    nb.fit(X_train, y_train)

    # Evaluate on validation
    y_pred_val = nb.predict(X_val)
    print("=== Validation (Pitch-Only NB) ===")
    print("Accuracy:", accuracy_score(y_val, y_pred_val))
    print(classification_report(y_val, y_pred_val, zero_division=0))

    # Evaluate on test
    y_pred_test = nb.predict(X_test)
    print("\n=== Test (Pitch-Only NB) ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_test))
    print(classification_report(y_test, y_pred_test, zero_division=0))

if __name__ == '__main__':
    main()
