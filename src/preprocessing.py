import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import pandas as pd

def basic_preprocess(features):
    features = features.drop(columns=[('feature','statistics')])

    corr = features.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]

    features = features.drop(columns=to_drop)

    return features


def scale_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    return X_scaled


def apply_pca(X, n_components):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    X_pca = pd.DataFrame(X_pca, index=X.index)
    return X_pca


def remove_outliers(X, contamination=0.02):
    iso = IsolationForest(contamination=contamination, random_state=42)
    labels = iso.fit_predict(X)
    mask = labels == 1
    return X[mask]