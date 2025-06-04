# baseline_lr.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

def main():
    df_train = pd.read_csv('train_set.csv')
    df_val   = pd.read_csv('val_set.csv')
    df_test  = pd.read_csv('test_set.csv')

    # All feature columns except 'filename' and 'composer'
    feat_cols = [c for c in df_train.columns if c not in ('filename','composer')]

    X_train = df_train[feat_cols]
    y_train = df_train['composer']
    X_val   = df_val[feat_cols]
    y_val   = df_val['composer']
    X_test  = df_test[feat_cols]
    y_test  = df_test['composer']

    # Pipeline: StandardScaler → LogisticRegression (multinomial)
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, C=1.0)
    )
    pipeline.fit(X_train, y_train)

    # Validate
    y_pred_val = pipeline.predict(X_val)
    print("=== Validation (All-Features LR) ===")
    print("Accuracy:", accuracy_score(y_val, y_pred_val))
    print(classification_report(y_val, y_pred_val, zero_division=0))

    # Test
    y_pred_test = pipeline.predict(X_test)
    print("\n=== Test (All-Features LR) ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_test))
    print(classification_report(y_test, y_pred_test, zero_division=0))

if __name__ == '__main__':
    main()
