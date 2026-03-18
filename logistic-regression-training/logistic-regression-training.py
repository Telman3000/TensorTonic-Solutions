import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    b = 0.0
    D = X.shape[1]
    N = X.shape[0]
    w = np.zeros(D)

    for _ in range(steps):
        z = X @ w + b
        p = _sigmoid(z)
        error = p - y
        dw = X.T @ error / N   #1/N​*∑​_i (p_i​−y_i​)x_ij​
        db = np.mean(error)    #db = Σ error_i / N
        w = w - lr * dw
        b = b - lr * db
    return w, b