import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report

import numpy as np
from sklearn.tree import DecisionTreeClassifier
np.random.seed(42)

N_ESTIMATORS = 100
MAX_DEPTH = 3
SUBSPACE_DIM = 2

class random_forest(object):
    def __init__(self, n_estimators: int, max_depth: int, subspaces_dim: int, random_state: int):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.subspaces_dim = subspaces_dim
        self.random_state = random_state
        self.estimators = []

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)  # инициализация генератора случайных чисел
        for i in range(self.n_estimators):
            # Использование Bootstrap Aggregating для случайного выбора объектов для текущего дерева
            idxs = rng.choice(X.shape[0], X.shape[0])
            X_subset, y_subset = X[idxs], y[idxs]

            # Создание и обучение дерева решений с заданными параметрами
            clf = DecisionTreeClassifier(max_depth=self.max_depth)
            clf.fit(X_subset, y_subset)
            self.estimators.append(clf)

    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            votes = {}  # Словарь для подсчета голосов деревьев за каждый класс
            for clf in self.estimators:
                pred = clf.predict([X[i]])[0]  # Предсказание метки на текущем дереве
                if pred not in votes:
                    votes[pred] = 1
                else:
                    votes[pred] += 1
            y_pred.append(max(votes, key=votes.get))  # Выбор метки с наибольшим количеством голосов
        return np.array(y_pred)
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
clf = random_forest(n_estimators=100, max_depth=3, subspaces_dim=2, random_state=1)
clf.fit(X, y)
# Прогнозируем метки классов
y_pred = clf.predict(X)
print(classification_report(y, y_pred))