import pandas as pd

def load_features(path):

    features = pd.read_csv(path, header=[0, 1], low_memory=False)

    features = features.iloc[2:]
    features.index = features.index.astype(int)
    features.index.name = 'track_id'

    return features