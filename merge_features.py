# merge_features.py

import pandas as pd

def main():
    # Load each feature CSV
    df_pitch  = pd.read_csv('pitch_features.csv')
    df_chord  = pd.read_csv('chord_features.csv')
    df_rhythm = pd.read_csv('rhythm_features.csv')

    # Merge them on ['filename','composer']
    df = df_pitch.merge(df_chord, on=['filename', 'composer'], how='inner')
    df = df.merge(df_rhythm, on=['filename', 'composer'], how='inner')

    # Check for any dropped rows
    total_rows = len(df_pitch)
    merged_rows = len(df)
    if merged_rows < total_rows:
        print(f"Warning: {total_rows - merged_rows} rows were dropped during merge. "
              "Check that all filenames/composer combos appear in each CSV.")
    else:
        print(f"All {merged_rows} rows merged successfully into all_features.csv")

    # Save the merged DataFrame
    df.to_csv('all_features.csv', index=False)
    print(f"Saved all_features.csv with shape {df.shape}")

if __name__ == '__main__':
    main()
