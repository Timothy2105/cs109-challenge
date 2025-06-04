# In your joint_generative.py (or a new tuning script)

import numpy as np
from sklearn.metrics import accuracy_score

# Assume you’ve already:
#  - loaded X_train, y_train, X_val, y_val, X_test, y_test, f_train, f_val, f_test
#  - estimated alpha_dict, prior_pitch from training
#  - computed beta-smoothed chord pi_dict, A_dict for β=10
#  - loaded chord_seqs, chord_to_idx, composers, prior_chord

# Precompute “pitch-only” log-likelihoods on validation for every (i,c):
logp_pitch_val = {}   # dict: i -> dict(composer -> log P_pitch(n_i | c))
for i, fn in enumerate(f_val):
    n_vec = X_val[i]
    logp_pitch_val[i] = {}
    for c in composers:
        logp_pitch_val[i][c] = log_dirichlet_multinomial(n_vec, alpha_dict[c]) \
                             + np.log(prior_pitch[c])

# Precompute “chord-only” log-likelihoods on validation for every (i,c):
logp_chord_val = {}   # dict: i -> dict(composer -> log P_chord(h_i | c))
for i, fn in enumerate(f_val):
    seq = chord_seqs.get(fn, [])
    logp_chord_val[i] = {}
    for c in composers:
        logp_chord_val[i][c] = log_markov_score(seq, pi_dict[c], A_dict[c],
                                                 chord_to_idx, len(chord_to_idx)) \
                             + np.log(prior_chord[c])  # if you want to add prior again

# Now do a simple grid search on λ
lambdas = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
best_lambda = None
best_acc = 0.0

for lam in lambdas:
    y_pred = []
    for i in range(len(f_val)):
        best_score = -np.inf
        best_c = None
        for c in composers:
            score = logp_pitch_val[i][c] + lam * logp_chord_val[i][c]
            # (Note: We already added P(c) into each term above; if you prefer to
            #  add P(c) only once, remove the + np.log(prior) from one of them.)
            if score > best_score:
                best_score = score
                best_c = c
        y_pred.append(best_c)
    acc = accuracy_score(y_val, y_pred)
    print(f"λ = {lam:<4} →  Val accuracy = {acc:.3f}")
    if acc > best_acc:
        best_acc = acc
        best_lambda = lam

print(f"\nBest λ on validation = {best_lambda}  with accuracy = {best_acc:.3f}")
