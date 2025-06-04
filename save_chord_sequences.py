# save_chord_sequences.py

import os
import json
import pandas as pd
from extract_chords import get_chord_sequence  # assumes get_chord_sequence is in extract_chords.py

def main():
    labels = pd.read_csv('labels.csv')  # contains all files you care about
    chord_seqs = {}

    for _, row in labels.iterrows():
        fn = row['filename']
        composer = row['composer']
        midi_path = os.path.join('data', composer, fn)  # or 'excerpts' if that's your folder
        seq = get_chord_sequence(midi_path)
        chord_seqs[fn] = seq

    # Save to JSON so we can use it for training/val/test
    with open('chord_sequences.json', 'w') as f:
        json.dump(chord_seqs, f)

    print(f"Saved {len(chord_seqs)} chord‐sequence entries to chord_sequences.json")

if __name__ == '__main__':
    main()
