import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pylab

np.random.seed(16)
x1 = 2*np.random.random(100) - 1
y1 = x1 + np.random.randn(100) * 200
z1 = x1**2 - y1**2 + x1*y1

class KNN_classifier:
    def __init__(self, n_neighbors: int, **kwargs):
        self.K = n_neighbors
    def euclidean_dist(self, array1, array2):
        array1=np.array(array1)
        array2=np.array(array2)
        return np.linalg.norm(array1-array2)
    def k_neighbors(self, test_row):
        dists = []
        for i in range(len(self.X_train)):
            dist = self.euclidean_dist(test_row, self.X_train[i])
            dists.append((dist, self.y_train[i]))
        dists.sort()
        return dists[:self.K]
    def get_nn(self):
        self.X_train = np.array(self.X_train)
        self.X_test= np.array(self.X_test)
        self.y_train = np.array(self.y_train)
        neib = []
        for j in range(len(self.X_test)):
            neib.append(self.k_neighbors(self.X_test[j]))
        return neib
    def vote_count(self, st):
        st_count = dict()
        for element in st:
          if element in st_count:
            st_count[element] +=1
          else:
            st_count[element] =1
        return st_count

    def fit(self, x: np.array, y: np.array):
        self.X_train = x
        self.y_train = y
        pass

    def predict(self, x: np.array):
        self.X_test = x
        nbr = self.get_nn()
        predictions = []
        for row in nbr:
          dist, labels = zip(*row)
          label_dict = self.vote_count(labels)
          predictions.append(max(label_dict, key = label_dict.get))
        predictions = np.array(predictions)
        return predictions
X=[]
y=[]
color=['b','g','r','y','m']
color1=[]
for i in range(100):
    X.append([x1[i],y1[i],z1[i]])
    col=np.random.randint(0,4)
    y.append(col)
    color1.append(color[col])
knn = KNN_classifier(n_neighbors=5)
knn.fit(X, y)
x_test=[]
x2 = 2*np.random.random(3) - 1
y2 = x2 + np.random.randn(3) * 200
z2 = x2**2 - y2**2 + x2*y2
for i in range(3):
    x_test.append([x2[i],y2[i],z2[i]])
k1=knn.predict(x_test)
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X, y)
k2=neigh.predict(x_test)
color2=[]
color3=[]
for i in range(3):
    color2.append(color[k1[i]])
    color3.append(color[k2[i]])
print(k1,k2)
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter3D(x1, y1, z1, c=color1)
ax.scatter3D(x2, y2, z2, c=color2, marker='^')
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.scatter3D(x1, y1, z1, c=color1)
ax.scatter3D(x2, y2, z2, c=color3, marker='^')
plt.show()
