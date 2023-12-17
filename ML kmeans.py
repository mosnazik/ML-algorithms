import numpy as np
from collections import defaultdict

class KMeans:
    def __init__(self, K, init):
        self.K = K
        self.centroids = init

    def fit(self, X):
        n_samples, n_features = X.shape
        centroids = self.centroids
        while True:
            clusters = defaultdict(list)
            for i, sample in enumerate(X):
                distances = [np.linalg.norm(sample - centroid) for centroid in centroids]
                closest_centroid = np.argmin(distances)
                clusters[closest_centroid].append(i)

            prev_centroids = centroids
            centroids = np.zeros((self.K, n_features))
            for cluster_idx, cluster in clusters.items():
                centroid = np.mean(X[cluster], axis=0)
                centroids[cluster_idx] = centroid

            diff = centroids - prev_centroids
            if np.all(diff < 0.001):
                break

        self.centroids = centroids

    def predict(self, X):
        clusters = defaultdict(list)
        for i, sample in enumerate(X):
            distances = [np.linalg.norm(sample - centroid) for centroid in self.centroids]
            closest_centroid = np.argmin(distances)
            clusters[closest_centroid].append(i)

        y_pred = np.zeros(len(X))
        for cluster_idx, cluster in clusters.items():
            for sample_idx in cluster:
                y_pred[sample_idx] = cluster_idx

        return y_pred