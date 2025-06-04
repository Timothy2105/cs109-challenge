# chord_markov_train.py

import json
import numpy as np
import pandas as pd

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)

# -----------------------------------------------------------------------------
# Step A: Count raw "initial" and "transition" occurrences on TRAINING split
# -----------------------------------------------------------------------------
def count_chord_transitions(chord_seqs, chord_to_idx, composers, train_csv):
    """
    Returns two dictionaries:
      N_init[c]  = length-H vector where N_init[c][i] = # times composer c's training
                     excerpts begin with vocab[i].
      N_trans[c] = H×H matrix where N_trans[c][i,j] = # times composer c's training
                   chord sequence has v_i → v_j.
    """
    df_train = pd.read_csv(train_csv)
    H = len(chord_to_idx)

    # Initialize counts
    N_init = {c: np.zeros(H, dtype=int) for c in composers}
    N_trans = {c: np.zeros((H, H), dtype=int) for c in composers}

    for _, row in df_train.iterrows():
        fn = row['filename']
        c = row['composer']
        seq = chord_seqs.get(fn, [])

        if not seq:
            continue

        # Count initial chord
        first = seq[0]
        if first in chord_to_idx:
            i0 = chord_to_idx[first]
            N_init[c][i0] += 1

        # Count transitions
        for t in range(len(seq) - 1):
            h_prev = seq[t]
            h_next = seq[t + 1]
            if h_prev in chord_to_idx and h_next in chord_to_idx:
                i = chord_to_idx[h_prev]
                j = chord_to_idx[h_next]
                N_trans[c][i, j] += 1

    return N_init, N_trans

# -----------------------------------------------------------------------------
# Step B: Smooth with Dirichlet (beta) to get probabilities π[c] and A[c]
# -----------------------------------------------------------------------------
def smooth_and_normalize(N_init, N_trans, beta=1.0):
    """
    Given raw counts N_init[c] and N_trans[c], apply Dirichlet smoothing:
      π[c][i]  = (beta + N_init[c][i]) / (H*beta + sum_j N_init[c][j])
      A[c][i,j] = (beta + N_trans[c][i,j]) / (H*beta + sum_k N_trans[c][i,k])

    Returns two dictionaries:
      pi[c] : length-H numpy array of initial‐chord probabilities
      A[c]  : H×H numpy array of transition probabilities
    """
    composers = list(N_init.keys())
    H = N_init[composers[0]].shape[0]

    pi = {}
    A  = {}

    for c in composers:
        # Smooth and normalize initial‐chord counts
        counts0 = N_init[c]
        denom0 = H * beta + np.sum(counts0)
        pi_c = (counts0 + beta) / denom0
        pi[c] = pi_c.tolist()   # convert to list for JSON‐friendliness

        # Smooth and normalize transition counts row by row
        A_c = np.zeros((H, H))
        for i in range(H):
            row_counts = N_trans[c][i, :]
            denom_i = H * beta + np.sum(row_counts)
            A_c[i, :] = (row_counts + beta) / denom_i
        A[c] = A_c.tolist()     # convert to nested lists

    return pi, A

# -----------------------------------------------------------------------------
# Main entry point: load, count, smooth, and save
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import sys

    if len(sys.argv) not in (3, 4):
        print("Usage: python chord_markov_train.py <chord_seq_json> <train_csv> [beta]")
        print("  <chord_seq_json> = 'chord_sequences.json'")
        print("  <train_csv>      = 'train_set.csv'")
        print("  [beta]           = optional Dirichlet smoothing constant (default 1.0)")
        sys.exit(1)

    chord_seq_json = sys.argv[1]  # e.g. 'chord_sequences.json'
    train_csv      = sys.argv[2]  # e.g. 'train_set.csv'
    beta           = float(sys.argv[3]) if len(sys.argv) == 4 else 1.0

    # 1) Load chord sequences from JSON
    print(f"Loading chord sequences from {chord_seq_json} ...")
    chord_seqs = load_json(chord_seq_json)  # dict: filename -> [list of chord labels]

    # 2) Build vocabulary and mapping from TRAINING set
    #    We assume you previously ran 'build_and_save_vocab.py' or have:
    #      - chord_vocab.json  (list of all H chord labels)
    #      - chord_to_idx.json (mapping chord label -> index 0..H-1)
    #
    #    If you haven't saved them yet, just load those here:
    with open('chord_to_idx.json', 'r') as f:
        chord_to_idx = json.load(f)
    composers = pd.read_csv(train_csv)['composer'].unique().tolist()
    H = len(chord_to_idx)

    print(f"Found {len(composer := composers)} composers: {composers}")
    print(f"Vocabulary size (H): {H} chords.")

    # 3) Count raw initial & transition occurrences on TRAINING set
    print(f"Counting chord occurrences on {train_csv} ...")
    N_init, N_trans = count_chord_transitions(chord_seqs, chord_to_idx, composers, train_csv)

    # 4) Apply Dirichlet smoothing to get probabilities
    print(f"Smoothing with β = {beta} and normalizing ...")
    pi_dict, A_dict = smooth_and_normalize(N_init, N_trans, beta=beta)

    # 5) Save the model parameters (pi and A) to disk so you can load them later
    save_json(pi_dict, 'chord_pi.json')
    save_json(A_dict, 'chord_A.json')

    print(f"Saved initial‐chord distributions to 'chord_pi.json'")
    print(f"Saved transition matrices to 'chord_A.json'")
    print("Done.")
