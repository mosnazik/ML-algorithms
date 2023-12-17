import numpy as np
from sklearn.datasets import load_digits
np.random.seed(42)
# Функция подсчета градиента
def gradient(y_true: int, y_pred: float, x: np.array) -> np.array:
    grad = []
    for i in range(len(x)):
        grad.append(x[i]*((1-y_true)*y_pred-y_true*(1-y_pred)))
    grad.append((1-y_true)*y_pred-y_true*(1-y_pred))
    return np.array(grad)


# Функция обновления весов
def update(alpha: np.array, gradient: np.array, lr: float)-> np.array:
    alpha_new = []
    print(gradient)
    for i in range(len(alpha)):
        alpha_new.append(alpha[i]-lr*gradient[i])
    return np.array(alpha_new)


# функция тренировки модели
def train(
    alpha0: np.array, x_train: np.array, y_train: np.array, lr: float, num_epoch: int
):
    alpha = alpha0.copy()
    for epo in range(num_epoch):
        for i, x in enumerate(x_train):
            y_pred=alpha[len(alpha)-1]
            for j in range(len(x)):
                y_pred=y_pred+alpha[j]*x[j]
            sig=1 / (1 + np.exp(-y_pred))
            alpha=update(alpha,gradient(y_train[i],sig,x),lr)
    return alpha
#X, y = load_digits(return_X_y=True)
#print(len(y))
#print(X[0])
x1=np.array([[1,2]])
w=np.array([2,-1,0])
y1=np.array([1])
print(train(w,x1,y1,1.0,1))