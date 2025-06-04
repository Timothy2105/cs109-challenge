# extract_pitch_counts.py

import os
import sys
import numpy as np
import pandas as pd
from music21 import converter

def get_raw_pitch_counts(midi_path):
    """
    Load a MIDI via music21, flatten it, and return a length‐12 integer array
    counting how many notes fall into each pitch‐class (0=C, …, 11=B).
    """
    score = converter.parse(midi_path).flatten()
    notes = score.getElementsByClass('Note')
    hist = np.zeros(12, dtype=int)
    for n in notes:
        pc = n.pitch.pitchClass
        hist[pc] += 1
    return hist

def build_counts_df(excerpt_dir, labels_csv):
    """
    For each (filename, composer) in labels_csv, compute raw pitch counts
    (length‐12 integer vector) and return a DataFrame:
      [filename, composer, count_0, count_1, …, count_11]
    """
    labels = pd.read_csv(labels_csv)
    rows = []
    for _, row in labels.iterrows():
        fn = row['filename']
        composer = row['composer']
        midi_path = os.path.join(excerpt_dir, composer, fn)
        if not os.path.isfile(midi_path):
            raise FileNotFoundError(f"Missing file: {midi_path}")
        counts = get_raw_pitch_counts(midi_path)
        d = {'filename': fn, 'composer': composer}
        for j in range(12):
            d[f'count_{j}'] = int(counts[j])
        rows.append(d)
    df_counts = pd.DataFrame(rows)
    return df_counts

if __name__ == '__main__':
    """
    Usage:
      python extract_pitch_counts.py <excerpt_dir> <labels_csv>

    Example:
      python extract_pitch_counts.py data labels.csv
    """
    if len(sys.argv) != 3:
        print("Usage: python extract_pitch_counts.py <excerpt_dir> <labels_csv>")
        sys.exit(1)

    excerpt_dir = sys.argv[1]   # e.g. 'data' or 'excerpts'
    labels_csv  = sys.argv[2]   # 'labels.csv'

    df_counts = build_counts_df(excerpt_dir, labels_csv)
    output_csv = 'pitch_counts.csv'
    df_counts.to_csv(output_csv, index=False)
    print(f"Saved {len(df_counts)} rows to {output_csv}")
