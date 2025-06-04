# make env
python -m venv venv
source venv/bin/activate

# install packages
pip install music21 pandas scikit-learn scipy

# setup
python3 make_labels.py
python3 extract_pitch_counts.py data labels.csv
python extract_chords.py data labels.csv
python extract_rhythm.py data labels.csv
python merge_features.py
python generative_pitch_counts.py
python save_chord_sequences.py
python chord_markov_build.py
python build_and_save_vocab.py
python chord_markov_train.py chord_sequences.json train_set.csv 10.0
python joint_pitch_chord_final.py
