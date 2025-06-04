import pandas as pd
import matplotlib.pyplot as plt

# 1) Load the pitch‐counts CSV
pitch_df = pd.read_csv('pitch_counts.csv')

# 2) Identify all composers
composers = pitch_df['composer'].unique()

# 3) Define labels for the 12 pitch classes
pitch_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# 4) For each composer, sum up their pitch counts and plot a bar chart
for composer in composers:
    subset = pitch_df[pitch_df['composer'] == composer]
    # columns 2..13 correspond to count_0 ... count_11
    sum_counts = subset.iloc[:, 2:].sum()  

    plt.figure(figsize=(8, 4))
    plt.bar(pitch_labels, sum_counts)
    plt.title(f'Pitch-Class Distribution for {composer.capitalize()}')
    plt.xlabel('Pitch Class')
    plt.ylabel('Total Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
