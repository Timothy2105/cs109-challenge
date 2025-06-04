import pandas as pd
import matplotlib.pyplot as plt

# 1) Load the chord‐features CSV
chord_df = pd.read_csv('chord_features.csv')

# 2) Identify all composers
composers = chord_df['composer'].unique()

# 3) Extract chord column names (everything after 'filename' and 'composer')
chord_labels = chord_df.columns[2:].tolist()

# 4) For each composer, sum up their chord counts and plot a bar chart
for composer in composers:
    subset = chord_df[chord_df['composer'] == composer]
    sum_chords = subset[chord_labels].sum()

    plt.figure(figsize=(12, 4))
    plt.bar(chord_labels, sum_chords)
    plt.title(f'Chord Frequency Distribution for {composer.capitalize()}')
    plt.xlabel('Chord')
    plt.ylabel('Total Count')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
