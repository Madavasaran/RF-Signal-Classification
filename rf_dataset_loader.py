import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_dataset(filepath, selected_mods=None, snr_threshold=None):
    with open(filepath, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    X, labels = [], []
    for (mod, snr), signal in data.items():
        if selected_mods and mod not in selected_mods:
            continue
        if snr_threshold is not None and snr < snr_threshold:
            continue
        for x in signal:
            X.append(x)
            labels.append(mod)

    X = np.array(X)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    return train_test_split(X, y, test_size=0.2, random_state=42), le