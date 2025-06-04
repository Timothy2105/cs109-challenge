# extract_rhythm.py

import os
import sys
import numpy as np
import pandas as pd
from music21 import converter

def get_rhythm_features(midi_path):
    """
    Parse the MIDI, flatten it, extract every Note’s duration (in quarter-length units)
    and its offset. Build:
      - A 5-bin normalized histogram over durations [0.25, 0.5, 1, 2, 4] (i.e. 16th, 8th, quarter, half, whole).
      - Mean & variance of all durations.
      - Mean & variance of inter-onset intervals (IOI).
    Returns a dict with keys:
      rh_bin_16th, rh_bin_8th, rh_bin_qtr, rh_bin_half, rh_bin_whole,
      rh_mean_dur, rh_var_dur, rh_mean_ioi, rh_var_ioi
    """
    score = converter.parse(midi_path).flat
    notes = score.getElementsByClass('Note')
    durations = [n.duration.quarterLength for n in notes]
    if len(durations) == 0:
        # If no notes, just return zeros
        return {
            'rh_bin_16th': 0.0, 'rh_bin_8th': 0.0, 'rh_bin_qtr': 0.0,
            'rh_bin_half': 0.0, 'rh_bin_whole': 0.0,
            'rh_mean_dur': 0.0, 'rh_var_dur': 0.0,
            'rh_mean_ioi': 0.0, 'rh_var_ioi': 0.0
        }

    # 5-bin histogram over durations
    bins = [0.25, 0.5, 1.0, 2.0, 4.0]  # i.e., 16th, 8th, quarter, half, whole
    hist, _ = np.histogram(durations, bins=bins + [np.inf], density=True)

    mean_dur = float(np.mean(durations))
    var_dur = float(np.var(durations))

    # Inter-onset intervals (IOI)
    onsets = sorted([n.offset for n in notes])
    ioi = np.diff(onsets) if len(onsets) > 1 else [0.0]
    mean_ioi = float(np.mean(ioi))
    var_ioi = float(np.var(ioi))

    return {
        'rh_bin_16th': float(hist[0]),
        'rh_bin_8th':  float(hist[1]),
        'rh_bin_qtr':  float(hist[2]),
        'rh_bin_half': float(hist[3]),
        'rh_bin_whole':float(hist[4]),
        'rh_mean_dur': mean_dur,
        'rh_var_dur':  var_dur,
        'rh_mean_ioi': mean_ioi,
        'rh_var_ioi':  var_ioi
    }

def build_rhythm_df(excerpt_dir, labels_csv):
    """
    Read labels_csv, and for each (filename, composer), compute that MIDI’s
    rhythm‐feature dict (via get_rhythm_features). Return a DataFrame with:
       [ filename | composer | rh_bin_16th | … | rh_var_ioi ]
    """
    labels = pd.read_csv(labels_csv)
    rows = []
    
    for _, row in labels.iterrows():
        fn = row['filename']
        composer = row['composer']
        midi_path = os.path.join(excerpt_dir, composer, fn)

        if not os.path.isfile(midi_path):
            raise FileNotFoundError(f"Missing file: {midi_path}")

        feats = get_rhythm_features(midi_path)
        d = {'filename': fn, 'composer': composer}
        d.update(feats)
        rows.append(d)

    df_rhythm = pd.DataFrame(rows)
    return df_rhythm

if __name__ == '__main__':
    """
    Usage:
      python extract_rhythm.py <excerpt_dir> <labels_csv>

    Where:
      <excerpt_dir> = 'data' or 'excerpts'
      <labels_csv>  = 'labels.csv'
    """
    if len(sys.argv) != 3:
        print("Usage: python extract_rhythm.py <excerpt_dir> <labels_csv>")
        sys.exit(1)

    excerpt_dir = sys.argv[1]
    labels_csv = sys.argv[2]

    df_rhythm = build_rhythm_df(excerpt_dir, labels_csv)
    df_rhythm.to_csv('rhythm_features.csv', index=False)
    print(f"Saved {len(df_rhythm)} rows to rhythm_features.csv")
