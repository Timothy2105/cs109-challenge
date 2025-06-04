# generative_pitch.py

import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.metrics import accuracy_score, classification_report

def load_data(split_csv):
    """
    Loads a split (train/val/test). Returns:
      - X_counts: 2D numpy array of shape (num_examples, 12) with raw pitch-class counts
      - y:    array of composer labels (strings)
    """
    df = pd.read_csv(split_csv)
    # The 12 pitch-class columns are named 'pc_0', 'pc_1', …, 'pc_11'.
    pc_cols = [f'pc_{i}' for i in range(12)]
    # Multiply normalized frequencies by total notes? Actually, 
    # since we normalized earlier, we need raw counts. If your df has only normalized pc,
    # you might need to re‐read raw counts. But assuming 'pc_i' are counts (not frequencies):
    X_counts = df[pc_cols].values.astype(int)
    y = df['composer'].values
    return X_counts, y

def estimate_dirichlet_params(X_counts, y, alpha0=1.0):
    """
    Given training counts X_counts (n_samples x 12) and labels y, 
    compute alpha_c_j for each composer c ∈ {chopin, beethoven, haydn, mozart}. 
    We use the simple MAP: alpha_{c,j} = alpha0 + sum_{i:y[i]=c} X_counts[i,j].
    Returns:
      - alpha: dict mapping composer -> array of length 12
      - prior: dict mapping composer -> P(c) = fraction of training examples
    """
    composers = np.unique(y)
    alpha = {}
    prior = {}
    total_examples = len(y)

    for c in composers:
        idx = np.where(y == c)[0]
        # Sum pitch counts over all excerpts of composer c
        sum_counts = np.sum(X_counts[idx, :], axis=0)
        # α_{c,j} = α0 + sum_j
        alpha[c] = alpha0 + sum_counts

        # P(c) = (# examples of c) / (total examples)
        prior[c] = len(idx) / total_examples

    return alpha, prior

def log_dirichlet_multinomial_prob(counts, alpha_c):
    """
    Given a single excerpt's counts vector (length‐12) and α_c (length‐12),
    compute log P(counts | composer c) using:
    
      log P(n | α) 
      = log Γ(sum_j α_j) 
        − log Γ(sum_j α_j + N) 
        + sum_{j=0..11} [ log Γ(α_j + n_j) − log Γ(α_j) ],
        
    where N = sum_j n_j. We’ll use scipy.special.gammaln for stability.
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
    Classify each row in X_counts (n_samples x 12).
    Returns predicted labels.
    """
    composers = list(alpha.keys())
    n_samples = X_counts.shape[0]
    y_pred = []

    for i in range(n_samples):
        n_vec = X_counts[i, :]
        best_logprob = -np.inf
        best_c = None

        for c in composers:
            log_pc = np.log(prior[c])                       # log P(c)
            log_pnx = log_dirichlet_multinomial_prob(n_vec, alpha[c])  # log P(n | c)
            score = log_pc + log_pnx

            if score > best_logprob:
                best_logprob = score
                best_c = c

        y_pred.append(best_c)

    return np.array(y_pred)

def main():
    # 1. Load train/val/test sets (assuming train_set.csv, etc. exist)
    X_train, y_train = load_data('train_set.csv')
    X_val,   y_val   = load_data('val_set.csv')
    X_test,  y_test  = load_data('test_set.csv')

    # 2. Estimate Dirichlet parameters and priors from training data
    alpha0 = 1.0   # base concentration (Laplace smoothing)
    alpha, prior = estimate_dirichlet_params(X_train, y_train, alpha0=alpha0)
    print("Estimated Dirichlet α parameters for each composer (first few values):")
    for c in alpha:
        print("  ", c, ":", np.round(alpha[c][:5], 2), "… total α =", np.sum(alpha[c]))

    # 3. Classify validation set
    y_pred_val = classify_dirichlet_multinomial(X_val, alpha, prior)
    print("\n=== Validation (Dirichlet–Multinomial) ===")
    print("Accuracy:", accuracy_score(y_val, y_pred_val))
    print(classification_report(y_val, y_pred_val, zero_division=0))

    # 4. Classify test set
    y_pred_test = classify_dirichlet_multinomial(X_test, alpha, prior)
    print("\n=== Test (Dirichlet–Multinomial) ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_test))
    print(classification_report(y_test, y_pred_test, zero_division=0))

if __name__ == '__main__':
    main()
