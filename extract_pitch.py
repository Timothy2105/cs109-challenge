# extract_pitch.py
import os
import sys
import numpy as np
import pandas as pd
from music21 import converter

def get_pitch_histogram(midi_path):
    """
    Load a MIDI file via music21, flatten all notes, and return
    a length‐12 normalized histogram of pitch classes (0=C, …, 11=B).
    If no notes are found, returns a zero vector.
    """
    score = converter.parse(midi_path).flatten()
    notes = score.getElementsByClass('Note')
    hist = np.zeros(12, dtype=int)
    for n in notes:
        pc = n.pitch.pitchClass
        hist[pc] += 1

    total = hist.sum()
    if total > 0:
        return hist / total   # normalized frequencies
    else:
        return hist           # all zeros if no notes

def build_pitch_df(excerpt_dir, labels_csv):
    """
    Walk through each (filename, composer) in labels_csv,
    compute its 12‐dim pitch‐class histogram, and return a DataFrame:
        [ filename | composer | pc_0 | pc_1 | ... | pc_11 ]
    """
    # Read labels.csv (columns: filename,composer)
    labels = pd.read_csv(labels_csv)

    rows = []
    for _, row in labels.iterrows():
        fn = row['filename']
        composer = row['composer']
        midi_path = os.path.join(excerpt_dir, composer, fn)

        if not os.path.isfile(midi_path):
            raise FileNotFoundError(f"Expected file not found: {midi_path}")

        hist = get_pitch_histogram(midi_path)  # array of length 12
        d = {'filename': fn, 'composer': composer}
        for i in range(12):
            d[f'pc_{i}'] = float(hist[i])
        rows.append(d)

    df_pitch = pd.DataFrame(rows)
    return df_pitch

if __name__ == '__main__':
    """
    Usage:
        python extract_pitch.py <excerpt_dir> <labels_csv>

    Where:
      <excerpt_dir> is the folder containing subfolders for each composer
                    (e.g. 'data' or 'excerpts'),
      <labels_csv>  is the path to your labels.csv in the project root.

    Example:
        python extract_pitch.py data labels.csv
        python extract_pitch.py excerpts labels.csv
    """
    if len(sys.argv) != 3:
        print("Usage: python extract_pitch.py <excerpt_dir> <labels_csv>")
        sys.exit(1)

    excerpt_dir = sys.argv[1]    # e.g. 'data' or 'excerpts'
    labels_csv  = sys.argv[2]    # e.g. 'labels.csv'

    # Build the DataFrame of pitch‐class features
    df_pitch = build_pitch_df(excerpt_dir, labels_csv)

    # Save to CSV in project root
    output_csv = 'pitch_features.csv'
    df_pitch.to_csv(output_csv, index=False)
    print(f"Saved {len(df_pitch)} rows to {output_csv}")
