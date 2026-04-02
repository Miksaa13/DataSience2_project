from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ParameterGrid
import numpy as np

def tune_gmm(X):
    param_grid = ["full", "diag", "tied"]

    results = []

    for components in range(3, 20):
        for covs in param_grid:
            gmm = GaussianMixture(n_components=components,covariance_type=covs ,max_iter=300, n_init=3, random_state=42)
            gmm.fit(X)

            results.append({"components": components,
                        "covariance_type": covs,
                        "bic": gmm.bic(X),
                        "aic": gmm.aic(X)})

    return results

from sklearn.cluster import HDBSCAN

def tune_hdbscan(X):
    results = []

    for min_cluster_size in range(5, 51, 5):
        for min_samples in range(1, 15, 2):

            clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, copy=False, metric='manhattan')
            labels = clusterer.fit_predict(X)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_points = np.sum(labels == -1)
            noise_ratio = noise_points / len(labels)

            try:
                dbcv = clusterer.relative_validity_
            except:
                dbcv = -1

            adjusted_score = dbcv * (1 - noise_ratio)

            results.append({ "min_cluster_size": min_cluster_size,
                            "min_samples": min_samples,
                            "dbcv": dbcv,
                            "adjusted_score": adjusted_score,
                            "n_clusters": n_clusters,
                            "noise_ratio": noise_ratio
            })

    return results


from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import time


def tune_optics(X):
    results = []

    for min_samples in range(3,16,2):
        for xi in [0.03, 0.05, 0.07, 0.1]:
            for min_cluster_size in range(4,20,2):

                clusterer = OPTICS(min_samples=min_samples,xi=xi, min_cluster_size=min_cluster_size)
                start = time.time()
                try:
                    labels = clusterer.fit_predict(X)
                except:
                    continue
                print("time:", time.time() - start)

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                noise_points = np.sum(labels == -1)
                noise_ratio = noise_points / len(labels)

                mask = labels != -1
                if len(set(labels[mask])) > 1:
                    sil = silhouette_score(X[mask], labels[mask])
                else:
                    sil = -1

                adjusted_score = sil * (1 - noise_ratio)

                results.append({"min_samples": min_samples,
                                "xi": xi,
                                "min_cluster_size": min_cluster_size,
                                "silhouette": sil,
                                "adjusted_score": adjusted_score,
                                "n_clusters": n_clusters,
                                "noise_ratio": noise_ratio
                })

    return results



from sklearn.cluster import AgglomerativeClustering

def tune_agglomerative(X):
    results = []

    for n_clusters in range(3,15,1):
        for linkage in ["ward", "complete", "average"]:
            for metric in ["euclidean", "manhattan", "cosine"]:

                if linkage == "ward" and metric != "euclidean":
                    continue

                try:
                    clusterer = AgglomerativeClustering(
                        n_clusters=n_clusters,
                        linkage=linkage,
                        metric=metric
                    )

                    labels = clusterer.fit_predict(X)

                except:
                    continue

                if len(set(labels)) > 1:
                    sil = silhouette_score(X, labels)
                else:
                    sil = -1

                results.append({"n_clusters": n_clusters,
                                "linkage": linkage,
                                "metric": metric,
                                "silhouette": sil
                })

    return results