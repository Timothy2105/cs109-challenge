# extract_chords.py

import os
import sys
import pandas as pd
from music21 import converter, harmony, chord, pitch, note

# -----------------------------
# Part A: Utility to label a chord via music21 or fallback to triad‐matching
# -----------------------------
def label_chord_with_music21(ch, key_ctx):
    """
    Attempt to label a music21 Chord object using chordSymbolFromChord in the given key context.
    Returns something like 'I', 'V7', 'viio/V', etc. If it fails, return None.
    """
    try:
        rn = harmony.chordSymbolFromChord(ch, context=key_ctx)
        return str(rn.figure)
    except Exception:
        return None

def label_triad_fallback(ch):
    """
    Fallback: examine the pitch‐class set of the chord.
    If it matches a basic major or minor triad (in any transposition), label it accordingly:
      [C,E,G] -> 'C:maj'
      [A,C,E] -> 'A:min'
    Otherwise return None.
    """
    # Collect unique pitch classes
    pcs = sorted({p.pitchClass for p in ch.pitches})
    # Pattern for (0,4,7) = major; (0,3,7) = minor (relative to root = 0)
    triad_map = {
        (0, 4, 7): 'maj',
        (0, 3, 7): 'min',
    }
    for root in range(12):
        # Shift every pitch class by -root (mod 12) to see if it matches a pattern
        rel = tuple(sorted(((pc - root) % 12) for pc in pcs))
        if rel in triad_map:
            quality = triad_map[rel]
            # Convert integer root (0–11) to a note name (e.g. 0->“C”, 10->“Bb”)
            name = pitch.Pitch(root).name.replace('-', 'b')
            return f"{name}:{quality}"
    return None

# -----------------------------
# Part B: Revised get_chord_sequence
# -----------------------------
def get_chord_sequence(midi_path):
    """
    1) Parse a MIDI via music21.
    2) chordify() the entire score so all simultaneous notes become Chord objects.
    3) For each chord, try label_chord_with_music21. If that fails, try triad_fallback.
    Returns a list of chord labels (strings).
    """
    score = converter.parse(midi_path)

    # Chordify the entire score (merges all tracks/parts into Chord objects)
    chordified = score.chordify()

    # Estimate the key context once
    try:
        key_ctx = score.analyze('key')
    except Exception:
        key_ctx = None

    rn_list = []
    for elem in chordified.recurse().getElementsByClass('Chord'):
        # 1) Try music21’s Roman‐numeral label (requires a valid key_ctx)
        if key_ctx is not None:
            rn_label = label_chord_with_music21(elem, key_ctx)
            if rn_label is not None:
                rn_list.append(rn_label)
                continue

        # 2) Fallback: see if chord matches a basic triad (major/minor)
        triad_label = label_triad_fallback(elem)
        if triad_label is not None:
            rn_list.append(triad_label)
        # If both fail (e.g. seventh chord or non‐triadic cluster), skip it

    return rn_list

# -----------------------------
# Part C: Build the “Top N” chord vocabulary
# -----------------------------
def build_chord_vocab(excerpt_dir, labels_csv, top_n=30):
    """
    Read labels.csv, walk through each (filename, composer), get its chord sequence,
    count frequencies, and return the top_n most frequent chord labels.
    """
    labels = pd.read_csv(labels_csv)
    freq = {}

    for _, row in labels.iterrows():
        fn = row['filename']
        composer = row['composer']
        midi_path = os.path.join(excerpt_dir, composer, fn)
        if not os.path.isfile(midi_path):
            raise FileNotFoundError(f"Missing file: {midi_path}")

        seq = get_chord_sequence(midi_path)
        if len(seq) == 0:
            # Warn the user which file produced no chords
            print(f"Warning: no chords found in {composer}/{fn}")

        for rn in seq:
            freq[rn] = freq.get(rn, 0) + 1

    # Sort by descending frequency, then take the top_n labels
    sorted_chords = sorted(freq.items(), key=lambda x: -x[1])
    top_chords = [ch for ch, _ in sorted_chords[:top_n]]
    return top_chords

# -----------------------------
# Part D: Build a DataFrame of chord‐counts per file
# -----------------------------
def build_chord_df(excerpt_dir, labels_csv, top_chords):
    """
    For each (filename, composer) in labels.csv, extract its chord sequence,
    count occurrences of each chord in top_chords, and return a DataFrame:
       [ filename | composer | chord_<top_chords[0]> | ... | chord_<top_chords[-1]> ]
    """
    labels = pd.read_csv(labels_csv)
    rows = []

    for _, row in labels.iterrows():
        fn = row['filename']
        composer = row['composer']
        midi_path = os.path.join(excerpt_dir, composer, fn)
        if not os.path.isfile(midi_path):
            raise FileNotFoundError(f"Missing file: {midi_path}")

        seq = get_chord_sequence(midi_path)
        counts = {c: 0 for c in top_chords}
        for rn in seq:
            if rn in counts:
                counts[rn] += 1

        d = {'filename': fn, 'composer': composer}
        for c in top_chords:
            d[f'chord_{c}'] = counts[c]
        rows.append(d)

    df_chord = pd.DataFrame(rows)
    return df_chord

# -----------------------------
# Main entry point
# -----------------------------
if __name__ == '__main__':
    """
    Usage:
      python extract_chords.py <excerpt_dir> <labels_csv> [top_n]

    Where:
      <excerpt_dir> = 'data' or 'excerpts'
      <labels_csv>  = 'labels.csv'
      [top_n]       = optional: number of distinct chord labels to keep (default=30)
    """
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python extract_chords.py <excerpt_dir> <labels_csv> [top_n]")
        sys.exit(1)

    excerpt_dir = sys.argv[1]
    labels_csv  = sys.argv[2]
    top_n       = int(sys.argv[3]) if len(sys.argv) == 4 else 30

    top_chords = build_chord_vocab(excerpt_dir, labels_csv, top_n=top_n)
    print(f"Top {top_n} chords (Roman‐numerals or triad fallback): {top_chords}")

    df_chord = build_chord_df(excerpt_dir, labels_csv, top_chords)
    output_csv = 'chord_features.csv'
    df_chord.to_csv(output_csv, index=False)
    print(f"Saved {len(df_chord)} rows to {output_csv}")
