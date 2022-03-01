import pandas
import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
auto = pandas.read_csv("auto-mpg.csv")
def cat_to_num(data):
    categories = numpy.unique(data)
    features = {}
    for cat in categories:
        binary = (data == cat)
        features["%s=%s" % (data.name, cat)] = binary.astype("int")
    return pandas.DataFrame(features)
auto = auto.join(cat_to_num(auto['origin']))
auto = auto.drop('origin', axis=1)

auto_train = auto[:int(0.8*len(auto))]
auto_test = auto[int(0.8*len(auto)):]

auto[:5]
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(auto_train.drop('mpg', axis=1), auto_train["mpg"])
pred_mpg = reg.predict(auto_test.drop('mpg',axis=1))
plt.plot(auto_test.mpg, pred_mpg, 'o')
x = numpy.linspace(10,40,5)
plt.plot(x, x, '-');
plt.show()

X=auto_train.drop('mpg', axis=1)
y=auto_train["mpg"]
crossvalidation = KFold(n_splits=10, random_state=None, shuffle=False)
scores = cross_val_score(reg, X, y, cv=5).mean()
print(scores)