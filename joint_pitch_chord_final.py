# joint_pitch_chord_final.py

import json
import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.metrics import accuracy_score, classification_report

# -------------- Helper functions for pitch Dirichlet ----------------

def load_pitch_counts(counts_csv):
    df = pd.read_csv(counts_csv)
    count_cols = [f'count_{i}' for i in range(12)]
    X_counts = df[count_cols].values.astype(int)
    y = df['composer'].values
    filenames = df['filename'].values
    return X_counts, y, filenames

def split_stratified(X, y, filenames, train_frac=0.70, val_frac=0.15, test_frac=0.15, random_state=42):
    from sklearn.model_selection import train_test_split
    # First split train vs. temp
    X_train, X_temp, y_train, y_temp, f_train, f_temp = train_test_split(
        X, y, filenames,
        test_size=(1.0 - train_frac),
        stratify=y,
        random_state=random_state
    )
    # Split temp into val vs. test
    X_val, X_test, y_val, y_test, f_val, f_test = train_test_split(
        X_temp, y_temp, f_temp,
        test_size=test_frac / (val_frac + test_frac),
        stratify=y_temp,
        random_state=random_state
    )
    return (X_train, y_train, f_train), (X_val, y_val, f_val), (X_test, y_test, f_test)

def estimate_dirichlet_params(X_counts, y, alpha0=1.0):
    composers = np.unique(y)
    alpha = {}
    prior = {}
    total = len(y)
    for c in composers:
        idx = np.where(y == c)[0]
        sum_counts = np.sum(X_counts[idx, :], axis=0)
        alpha[c] = alpha0 + sum_counts
        prior[c] = len(idx) / total
    return alpha, prior

def log_dirichlet_multinomial(counts, alpha_c):
    alpha = alpha_c
    n = counts
    N = np.sum(n)
    sum_alpha = np.sum(alpha)
    term1 = gammaln(sum_alpha)
    term2 = gammaln(sum_alpha + N)
    term3 = np.sum(gammaln(alpha + n) - gammaln(alpha))
    return term1 - term2 + term3

# -------------- Helper function for chord Markov --------------------

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def log_markov_score(seq, pi_c, A_c, chord_to_idx, H):
    if len(seq) == 0:
        return -np.log(H)
    h0 = seq[0]
    if h0 in chord_to_idx:
        i0 = chord_to_idx[h0]
        logp = np.log(pi_c[i0])
    else:
        logp = -np.log(H)
    for t in range(1, len(seq)):
        prev = seq[t - 1]
        curr = seq[t]
        if prev in chord_to_idx and curr in chord_to_idx:
            i_prev = chord_to_idx[prev]
            i_curr = chord_to_idx[curr]
            logp += np.log(A_c[i_prev][i_curr])
        else:
            logp += -np.log(H)
    return logp

# -------------- Main: Joint classification on val + test ------------

def main():
    # 1) Load & split pitch counts
    X, y, filenames = load_pitch_counts('pitch_counts.csv')
    (X_train, y_train, f_train), (X_val, y_val, f_val), (X_test, y_test, f_test) = split_stratified(X, y, filenames)

    # 2) Estimate pitch Dirichlet params (use your chosen alpha0; e.g. alpha0=1.0)
    alpha0 = 1.0
    alpha_dict, prior_pitch = estimate_dirichlet_params(X_train, y_train, alpha0)

    # 3) Load chord Markov parameters (β = 10)
    chord_seqs    = load_json('chord_sequences.json')
    chord_to_idx  = load_json('chord_to_idx.json')
    composers     = sorted(load_json('chord_pi.json').keys())  # ['beethoven','chopin','haydn','mozart']
    pi_dict       = load_json('chord_pi.json')                 # composer -> [length H]
    A_dict        = load_json('chord_A.json')                  # composer -> [H×H nested lists]
    H = len(chord_to_idx)

    # 4) Build composer prior P(c) from train_set.csv (this should match prior_pitch keys)
    df_train = pd.read_csv('train_set.csv')
    total_train = len(df_train)
    prior_composer = {c: (df_train['composer']==c).sum() / total_train for c in composers}

    # ------- Classify a generic split (helper) -------
    def classify_joint(X_split, f_split, y_true_split):
        y_pred = []
        for i, fn in enumerate(f_split):
            n_vec = X_split[i, :]
            seq   = chord_seqs.get(fn, [])

            best_score = -np.inf
            best_c = None
            for c in composers:
                # log P(c)
                log_pc = np.log(prior_composer[c])

                # log P_pitch(n_vec | c)
                log_p_pitch = log_dirichlet_multinomial(n_vec, alpha_dict[c])

                # log P_chord(seq | c)
                log_p_chord = log_markov_score(seq, pi_dict[c], A_dict[c], chord_to_idx, H)

                # total joint score
                score = log_pc + log_p_pitch + log_p_chord
                if score > best_score:
                    best_score = score
                    best_c = c

            y_pred.append(best_c)

        y_pred = np.array(y_pred)
        acc = accuracy_score(y_true_split, y_pred)
        report = classification_report(y_true_split, y_pred, zero_division=0)
        return acc, report

    # 5) Evaluate on validation
    val_acc, val_report = classify_joint(X_val, f_val, y_val)
    print("=== Joint Generative on val_set.csv (β=10, α₀=1) ===")
    print("Accuracy:", val_acc)
    print(val_report)

    # 6) Evaluate on test
    test_acc, test_report = classify_joint(X_test, f_test, y_test)
    print("\n=== Joint Generative on test_set.csv (β=10, α₀=1) ===")
    print("Accuracy:", test_acc)
    print(test_report)


if __name__ == '__main__':
    main()
