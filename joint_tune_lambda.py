# joint_tune_lambda.py

import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------------------------------
# STEP 0: IMPORT YOUR EXISTING FUNCTIONS
# -------------------------------------------------------
# Adjust these import paths if your files are named differently.
# From generative_pitch_counts.py:
#   - load_counts_and_labels() → returns (X_counts, y, filenames)
#   - split_stratified()     → splits into train/val/test
#   - estimate_dirichlet_params()
#   - log_dirichlet_multinomial()

from generative_pitch_counts import (
    load_counts_and_labels,
    split_stratified,
    estimate_dirichlet_params,
    log_dirichlet_multinomial
)

# From chord_markov_loglik.py:
from chord_markov_classify import log_markov_score


# -------------------------------------------------------
# STEP 1: LOAD PITCH COUNTS & SPLIT INTO TRAIN/VAL/TEST
# -------------------------------------------------------
# load_counts_and_labels reads “pitch_counts.csv” and returns (X_counts, y, filenames)
X_counts, y_all, filenames_all = load_counts_and_labels('pitch_counts.csv')

# split_stratified splits (X_counts, y_all, filenames_all) into train/val/test
(X_train, y_train, f_train), (X_val, y_val, f_val), (X_test, y_test, f_test) = \
    split_stratified(X_counts, y_all, filenames_all,
                     train_frac=0.70, val_frac=0.15, test_frac=0.15,
                     random_state=42)

# -------------------------------------------------------
# STEP 2: ESTIMATE DIRICHLET‐MULTINOMIAL (PITCH) PARAMETERS ON TRAIN
# -------------------------------------------------------
alpha0 = 1.0
alpha_dict, prior_pitch = estimate_dirichlet_params(X_train, y_train, alpha0=alpha0)

# -------------------------------------------------------
# STEP 3: LOAD CHORD‐MARKOV PARAMETERS (β = 10)
# -------------------------------------------------------
with open('chord_pi.json', 'r') as f:
    pi_dict = {c: np.array(vals) for c, vals in json.load(f).items()}

with open('chord_A.json', 'r') as f:
    A_dict = {c: np.array(mat) for c, mat in json.load(f).items()}

with open('chord_to_idx.json', 'r') as f:
    chord_to_idx = json.load(f)

# Composer list & chord prior from train_set.csv
df_train = pd.read_csv('train_set.csv')
composers = sorted(df_train['composer'].unique().tolist())

total_train = len(df_train)
prior_chord = {c: (df_train['composer'] == c).sum() / total_train for c in composers}

# Load the chord sequences (filename -> ordered chord list)
with open('chord_sequences.json', 'r') as f:
    chord_seqs = json.load(f)

# -------------------------------------------------------
# STEP 4: PRECOMPUTE “PITCH‐ONLY” & “CHORD‐ONLY” LOG‐PROBS ON VALIDATION
# -------------------------------------------------------
# 4A) Pitch‐only on validation (including + log P(c))
logp_pitch_val = {}    # i -> { composer c -> log P_pitch(n_i | c) + log P(c) }
for i, fn in enumerate(f_val):
    n_vec = X_val[i]
    logp_pitch_val[i] = {}
    for c in composers:
        logp_pitch_val[i][c] = (
            log_dirichlet_multinomial(n_vec, alpha_dict[c])
            + np.log(prior_pitch[c])
        )

# 4B) Chord‐only on validation (including + log P(c))
logp_chord_val = {}    # i -> { c -> log P_chord(h_i | c) + log P(c) }
for i, fn in enumerate(f_val):
    seq = chord_seqs.get(fn, [])
    logp_chord_val[i] = {}
    for c in composers:
        logp_chord_val[i][c] = (
            log_markov_score(seq, pi_dict[c], A_dict[c], chord_to_idx, len(chord_to_idx))
            + np.log(prior_chord[c])
        )

# -------------------------------------------------------
# STEP 5: GRID‐SEARCH λ ON VALIDATION
# -------------------------------------------------------
lambdas = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
best_lambda = None
best_acc = -1.0

print("\nTuning λ (weight on chord term) over:", lambdas)
print("λ     |  Val Accuracy")
print("------------------------")
for lam in lambdas:
    y_pred = []
    for i in range(len(f_val)):
        best_score = -np.inf
        best_c = None
        for c in composers:
            # joint_score = pitch_term + λ * chord_term
            score = logp_pitch_val[i][c] + lam * logp_chord_val[i][c]
            if score > best_score:
                best_score = score
                best_c = c
        y_pred.append(best_c)

    acc = accuracy_score(y_val, y_pred)
    print(f"{lam:<5} |   {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_lambda = lam

print(f"\n→ Best λ on validation = {best_lambda}  (accuracy = {best_acc:.4f})\n")

# Detailed classification report on validation with best_lambda
y_pred_best = []
for i in range(len(f_val)):
    best_score = -np.inf
    best_c = None
    for c in composers:
        score = logp_pitch_val[i][c] + best_lambda * logp_chord_val[i][c]
        if score > best_score:
            best_score = score
            best_c = c
    y_pred_best.append(best_c)

print(f"=== Detailed report on val_set.csv  (λ = {best_lambda}) ===")
print(classification_report(y_val, y_pred_best, zero_division=0))

# -------------------------------------------------------
# STEP 6: PRECOMPUTE “PITCH‐ONLY” & “CHORD‐ONLY” LOG‐PROBS ON TEST
# -------------------------------------------------------
# 6A) Pitch‐only on test
logp_pitch_test = {}
for i, fn in enumerate(f_test):
    n_vec = X_test[i]
    logp_pitch_test[i] = {}
    for c in composers:
        logp_pitch_test[i][c] = (
            log_dirichlet_multinomial(n_vec, alpha_dict[c])
            + np.log(prior_pitch[c])
        )

# 6B) Chord‐only on test
logp_chord_test = {}
for i, fn in enumerate(f_test):
    seq = chord_seqs.get(fn, [])
    logp_chord_test[i] = {}
    for c in composers:
        logp_chord_test[i][c] = (
            log_markov_score(seq, pi_dict[c], A_dict[c], chord_to_idx, len(chord_to_idx))
            + np.log(prior_chord[c])
        )

# -------------------------------------------------------
# STEP 7: CLASSIFY TEST WITH best_lambda
# -------------------------------------------------------
y_pred_test = []
for i in range(len(f_test)):
    best_score = -np.inf
    best_c = None
    for c in composers:
        score = logp_pitch_test[i][c] + best_lambda * logp_chord_test[i][c]
        if score > best_score:
            best_score = score
            best_c = c
    y_pred_test.append(best_c)

print(f"\n=== Joint Final on test_set.csv  (λ = {best_lambda}) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test, zero_division=0))

