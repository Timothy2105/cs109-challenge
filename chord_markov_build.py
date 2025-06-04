# chord_markov_build.py

import json
import numpy as np
import pandas as pd
from collections import defaultdict

def load_chord_sequences(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)   # dict: filename -> [ chord1, chord2, … ]

def build_vocab_and_composer_indices(labels_csv, chord_seqs):
    """
    1) Read labels.csv → knows which composer each filename belongs to.
    2) Collect all chord labels from the training set → builds a sorted vocabulary.
    3) Return:
         - vocab: list of chord‐strings, size H
         - chord_to_idx: dict mapping chord -> index in [0..H-1]
         - composers: sorted list of composer names (e.g. ["beethoven","chopin",…])
         - file2composer: dict filename -> composer
    """
    df = pd.read_csv(labels_csv)
    composers = sorted(df['composer'].unique().tolist())

    # Collect chord labels that appear in ANY training file
    chord_set = set()
    file2composer = {}
    for _, row in df.iterrows():
        fn = row['filename']
        c = row['composer']
        file2composer[fn] = c
        seq = chord_seqs.get(fn, [])
        chord_set.update(seq)

    vocab = sorted(chord_set)
    chord_to_idx = {h: i for i, h in enumerate(vocab)}
    return vocab, chord_to_idx, composers, file2composer

# Example usage:
if __name__ == '__main__':
    chord_seqs = load_chord_sequences('chord_sequences.json')
    vocab, chord_to_idx, composers, file2composer = build_vocab_and_composer_indices('train_set.csv', chord_seqs)
    print("Vocabulary size H =", len(vocab))
    print("Example vocab entries:", vocab[:10])
    print("Composers:", composers)
