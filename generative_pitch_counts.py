# generative_pitch_counts.py

import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def load_counts_and_labels(counts_csv):
    """
    Loads pitch_counts.csv which has columns:
      filename, composer, count_0, count_1, ..., count_11.
    Returns:
      - X_counts: numpy array of shape (n_examples, 12) with integer counts
      - y:        array of composer labels (strings)
      - filenames: array of filenames (strings)
    """
    df = pd.read_csv(counts_csv)
    count_cols = [f'count_{i}' for i in range(12)]
    X_counts = df[count_cols].values.astype(int)
    y = df['composer'].values
    filenames = df['filename'].values
    return X_counts, y, filenames

def split_stratified(X, y, filenames, train_frac=0.70, val_frac=0.15, test_frac=0.15, random_state=42):
    """
    Stratified split into train/val/test. Returns:
      (X_train, y_train, f_train),
      (X_val,   y_val,   f_val),
      (X_test,  y_test,  f_test).
    """
    # First split train vs. temp (val+test)
    X_train, X_temp, y_train, y_temp, f_train, f_temp = train_test_split(
        X, y, filenames,
        test_size=(1.0 - train_frac),
        stratify=y,
        random_state=random_state
    )
    # Then split temp into val vs. test (half/half of temp)
    X_val, X_test, y_val, y_test, f_val, f_test = train_test_split(
        X_temp, y_temp, f_temp,
        test_size=test_frac / (val_frac + test_frac),
        stratify=y_temp,
        random_state=random_state
    )
    return (X_train, y_train, f_train), (X_val, y_val, f_val), (X_test, y_test, f_test)

def estimate_dirichlet_params(X_counts, y, alpha0=1.0):
    """
    For each composer c, estimate alpha_c as:
      alpha_{c,j} = alpha0 + sum_{i: y[i]==c} X_counts[i,j].
    Returns:
      - alpha: dict mapping composer -> array of length 12
      - prior: dict mapping composer -> fraction of training examples
    """
    composers = np.unique(y)
    alpha = {}
    prior = {}
    total_examples = len(y)

    for c in composers:
        idx_c = np.where(y == c)[0]
        # Sum raw counts over all excerpts of composer c
        sum_counts = np.sum(X_counts[idx_c, :], axis=0)
        alpha[c] = alpha0 + sum_counts
        prior[c] = len(idx_c) / total_examples

    return alpha, prior

def log_dirichlet_multinomial(counts, alpha_c):
    """
    Dirichlet-Multinomial log‐probability:
      log P(n | alpha) = log Γ(sum α) - log Γ(sum α + N)
                        + sum_j [ log Γ(α_j + n_j) - log Γ(α_j ) ]
    where N = sum_j n_j.
    """
    alpha = alpha_c
    n = counts
    N = np.sum(n)
    sum_alpha = np.sum(alpha)

    term1 = gammaln(sum_alpha)
    term2 = gammaln(sum_alpha + N)
    term3 = np.sum(gammaln(alpha + n) - gammaln(alpha))

    return term1 - term2 + term3

def classify_dirichlet_multinomial(X_counts, alpha, prior):
    """
    Classify each row in X_counts under the Dirichlet–Multinomial generative model.
    Returns predicted composer labels.
    """
    composers = list(alpha.keys())
    n_samples = X_counts.shape[0]
    y_pred = []

    for i in range(n_samples):
        n_vec = X_counts[i, :]
        best_score = -np.inf
        best_c = None
        for c in composers:
            log_pc = np.log(prior[c])
            log_pn = log_dirichlet_multinomial(n_vec, alpha[c])
            score = log_pc + log_pn
            if score > best_score:
                best_score = score
                best_c = c
        y_pred.append(best_c)

    return np.array(y_pred)

def main():
    # 1. Load raw counts and labels
    X, y, filenames = load_counts_and_labels('pitch_counts.csv')

    # 2. Stratified split into train / val / test
    (X_train, y_train, f_train), (X_val, y_val, f_val), (X_test, y_test, f_test) = split_stratified(
        X, y, filenames, train_frac=0.70, val_frac=0.15, test_frac=0.15, random_state=42
    )
    print(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

    # === BEGIN α₀‐SWEEP SECTION ===
    print("\nSweeping over alpha0 values on validation set:")
    for alpha0 in [0.1, 0.5, 1.0, 2.0, 5.0]:
        # Estimate Dirichlet parameters with this alpha0
        alpha_dict, prior_dict = estimate_dirichlet_params(X_train, y_train, alpha0=alpha0)

        # Classify the validation set
        y_val_pred = classify_dirichlet_multinomial(X_val, alpha_dict, prior_dict)

        # Compute validation accuracy
        val_acc = accuracy_score(y_val, y_val_pred)
        print(f"  alpha0 = {alpha0:>4} → val accuracy = {val_acc:.3f}")
    # === END α₀‐SWEEP SECTION ===

    # (Optional: pick the best alpha0 from above, then re-estimate and test on X_test)
    # For example, if α₀ = 0.5 was best, do:
    # alpha_best, prior_best = estimate_dirichlet_params(X_train, y_train, alpha0=0.5)
    # y_test_pred = classify_dirichlet_multinomial(X_test, alpha_best, prior_best)
    # print("\n=== Test (with best alpha0=0.5) ===")
    # print("Accuracy:", accuracy_score(y_test, y_test_pred))
    # print(classification_report(y_test, y_test_pred, zero_division=0))

if __name__ == '__main__':
    main()