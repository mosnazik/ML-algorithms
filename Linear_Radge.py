import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import PolynomialFeatures

# Данные сборки
import csv
with open('auto-mpg.csv') as f:
    r = csv.reader(f)
    cont = [row for row in r]
    X = [ [0]*1 for i in range(len(cont)-1) ]
    y=[0]*(len(cont)-1)
for i in range(len(cont)-1):
    X[i][0]=float(cont[i+1][0])
    y[i] = float(cont[i + 1][5])
lr = linear_model.LinearRegression()
lr.fit(X, y)
clf = Ridge(alpha=10000.0)
clf.fit(X, y)
poly_reg = PolynomialFeatures(degree=2)
X_poly=poly_reg.fit_transform(X)
lin_reg2 = linear_model.LinearRegression()
lin_reg2.fit(X_poly,y)
# Predict data of estimated models
line_y = lr.predict(X)
line_z = clf.predict(X)
lw = 2
plt.scatter(
    X, y, color="yellowgreen"
)
plt.plot(X, line_y, color="navy", linewidth=lw, label="Linear regressor")
plt.plot(X, line_z, color="green", linewidth=lw, label="Ridge regressor")
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.show()
print("R^2 = ",lr.score(X,y))
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
print("R^2_CV = ",cross_val_score(lr,X,y,cv=cv).mean())


