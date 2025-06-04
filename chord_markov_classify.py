import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def log_markov_score(seq, pi_c, A_c, chord_to_idx, H):
    """
    Compute  log P(seq | composer c) 
    for a given chord sequence `seq` = [h1, h2, …, hT].
    """
    if len(seq) == 0:
        return -np.log(H)

    # Initial‐chord
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
            logp += np.log(A_c[i_prev][i_curr])
        else:
            logp += -np.log(H)

    return logp

def classify_chord_markov(split_csv, chord_seq_json, chord_pi_json, chord_A_json, chord_to_idx_json):
    """
    For each excerpt in `split_csv` (val or test), compute
      log P(c) + log P_chord(seq | c)
    and pick the composer with highest total score.
    """
    # Load parameters
    chord_seqs   = load_json(chord_seq_json)   # filename -> [list of chord labels]
    pi_dict      = load_json(chord_pi_json)    # composer -> [list of length H]
    A_dict       = load_json(chord_A_json)     # composer -> H×H nested lists
    chord_to_idx = load_json(chord_to_idx_json)   # chord‐label -> index
    composers    = list(pi_dict.keys())
    H            = len(chord_to_idx)

    # Build composer prior P(c) from train_set.csv
    df_train = pd.read_csv('train_set.csv')
    total_train = len(df_train)
    prior = {c: sum(df_train['composer'] == c) / total_train for c in composers}

    # Classify each file in split_csv
    df_split = pd.read_csv(split_csv)
    y_true = df_split['composer'].values
    filenames = df_split['filename'].values

    y_pred = []
    for fn in filenames:
        seq = chord_seqs.get(fn, [])  # ordered chord list for this file

        best_score = -np.inf
        best_c = None
        for c in composers:
            log_pc    = np.log(prior[c])
            log_chord = log_markov_score(
                seq,
                np.array(pi_dict[c]),
                np.array(A_dict[c]),
                chord_to_idx,
                H
            )
            score = log_pc + log_chord
            if score > best_score:
                best_score = score
                best_c = c
        y_pred.append(best_c)

    print(f"\n=== Chord Markov on {split_csv} ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, zero_division=0))
    return y_pred, y_true
