# chord_markov_tune_beta.py

import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def count_chord_transitions(chord_seqs, chord_to_idx, composers, train_csv):
    """
    Count N_init[c] and N_trans[c] on the training split.
    """
    df_train = pd.read_csv(train_csv)
    H = len(chord_to_idx)

    N_init = {c: np.zeros(H, dtype=int) for c in composers}
    N_trans = {c: np.zeros((H, H), dtype=int) for c in composers}

    for _, row in df_train.iterrows():
        fn = row['filename']
        c  = row['composer']
        seq = chord_seqs.get(fn, [])

        if not seq:
            continue

        # Initial chord
        h0 = seq[0]
        if h0 in chord_to_idx:
            i0 = chord_to_idx[h0]
            N_init[c][i0] += 1

        # Transitions
        for t in range(len(seq) - 1):
            prev = seq[t]
            curr = seq[t + 1]
            if prev in chord_to_idx and curr in chord_to_idx:
                i_prev = chord_to_idx[prev]
                i_curr = chord_to_idx[curr]
                N_trans[c][i_prev, i_curr] += 1

    return N_init, N_trans

def smooth_and_normalize(N_init, N_trans, beta):
    """
    Apply Dirichlet(beta) smoothing to raw counts:
      π[c][i]  = (β + N_init[c][i]) / (Hβ + sum_j N_init[c][j])
      A[c][i,j] = (β + N_trans[c][i,j]) / (Hβ + sum_k N_trans[c][i,k])
    """
    composers = list(N_init.keys())
    H = N_init[composers[0]].shape[0]

    pi = {}
    A  = {}

    for c in composers:
        counts0 = N_init[c]
        denom0 = H * beta + np.sum(counts0)
        pi_c = (counts0 + beta) / denom0
        pi[c] = pi_c

        A_c = np.zeros((H, H))
        for i in range(H):
            row_counts = N_trans[c][i, :]
            denom_i = H * beta + np.sum(row_counts)
            A_c[i, :] = (row_counts + beta) / denom_i
        A[c] = A_c

    return pi, A

def log_markov_score(seq, pi_c, A_c, chord_to_idx, H):
    """
    Compute log P(seq | composer c) under the chord‐Markov model.
    """
    if len(seq) == 0:
        return -np.log(H)

    # Initial chord
    h0 = seq[0]
    if h0 in chord_to_idx:
        i0 = chord_to_idx[h0]
        logp = np.log(pi_c[i0])
    else:
        logp = -np.log(H)

    # Transitions
    for t in range(1, len(seq)):
        prev = seq[t - 1]
        curr = seq[t]
        if prev in chord_to_idx and curr in chord_to_idx:
            i_prev = chord_to_idx[prev]
            i_curr = chord_to_idx[curr]
            logp += np.log(A_c[i_prev, i_curr])
        else:
            logp += -np.log(H)

    return logp

def classify_chord_markov_split(chord_seqs, chord_to_idx, composers, prior, pi_dict, A_dict, split_csv):
    """
    Classify every file in split_csv using P(c) + log P_chord(seq | c).
    Returns (y_true, y_pred).
    """
    H = len(chord_to_idx)
    df_split = pd.read_csv(split_csv)
    filenames = df_split['filename'].values
    y_true = df_split['composer'].values

    y_pred = []
    for fn in filenames:
        seq = chord_seqs.get(fn, [])
        best_score = -np.inf
        best_c = None
        for c in composers:
            log_pc = np.log(prior[c])
            log_chord = log_markov_score(seq, pi_dict[c], A_dict[c], chord_to_idx, H)
            score = log_pc + log_chord
            if score > best_score:
                best_score = score
                best_c = c
        y_pred.append(best_c)

    return y_true, np.array(y_pred)

def main():
    # 1) Load chord sequences & chord_to_idx
    chord_seqs   = load_json('chord_sequences.json')
    chord_to_idx = load_json('chord_to_idx.json')
    H = len(chord_to_idx)

    # 2) Determine composers from training split
    df_train = pd.read_csv('train_set.csv')
    composers = sorted(df_train['composer'].unique().tolist())

    # 3) Compute composer prior P(c) from train_set.csv
    total_train = len(df_train)
    prior = {c: (df_train['composer'] == c).sum() / total_train for c in composers}

    # 4) Count raw N_init, N_trans on training split
    N_init, N_trans = count_chord_transitions(chord_seqs, chord_to_idx, composers, 'train_set.csv')

    # 5) Sweep over beta values
    betas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    print("\nSweeping β over:", betas)
    results = []

    for beta in betas:
        # Smooth & normalize to get π and A for this beta
        pi_dict, A_dict = smooth_and_normalize(N_init, N_trans, beta)

        # Classify validation set
        y_val_true, y_val_pred = classify_chord_markov_split(
            chord_seqs, chord_to_idx, composers, prior, pi_dict, A_dict, 'val_set.csv'
        )
        val_acc = accuracy_score(y_val_true, y_val_pred)
        results.append((beta, val_acc))

    # 6) Print summary
    print("\nβ    |  Validation Accuracy")
    print("---------------------------")
    for beta, acc in results:
        print(f"{beta:<4} |    {acc:.4f}")

    # 7) (Optional) Pick best β and show full classification report on validation
    best_beta, best_acc = max(results, key=lambda x: x[1])
    print(f"\nBest β on validation = {best_beta:.2f} → {best_acc:.4f}\n")

    # Smooth & normalize again with best_beta
    pi_best, A_best = smooth_and_normalize(N_init, N_trans, best_beta)

    # Print out classification report on val_set.csv
    y_val_true, y_val_pred = classify_chord_markov_split(
        chord_seqs, chord_to_idx, composers, prior, pi_best, A_best, 'val_set.csv'
    )
    print(f"=== Detailed report on val_set.csv (β = {best_beta:.2f}) ===")
    print(classification_report(y_val_true, y_val_pred, zero_division=0))

if __name__ == '__main__':
    main()
