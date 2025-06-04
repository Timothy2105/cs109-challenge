# make env
python -m venv venv
source venv/bin/activate

# install packages
pip install music21 pandas scikit-learn

# setup
python3 make_labels.py
python3 extract_pitch.py data labels.csv
python extract_chords.py data labels.csv
python extract_rhythm.py data labels.csv
python merge_features.py
