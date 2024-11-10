import numpy as np


def gradient_descent(
    X: np.array, W: np.array, Y: np.array, lr: float, regularize=False, reg_strength=0
):
    """Single step of gradient descent for Loss L= sum_i from(X_i W -y_i)^2"""
    loss_vals = []
    grads = []
    for i in range(X.shape[0]):
        x = X[i]
        y = Y[i]
        grad = 2 * (x.dot(W) - y) * x
        grads.append(grad)
        loss = np.sum((X.dot(W) - y) ** 2)
        loss_vals.append(loss)
    if regularize:
        grad += reg_strength * W
    W = W - lr * np.sum(grads, axis=0)

    return W, np.sum(loss_vals)


def main():
    X = [[-1, 2, 1, 1, -1], [-2, 1, -2, 0, 2], [1, 0, -2, -2, -1]]

    y = [5, 1, 1]
    w = [0, 0, 0, 0, 0]
    losses = []
    epoch = []
    for i in range(200):
        W, loss = gradient_descent(np.array(X), np.array(w), np.array(y), 0.02)
        w = W
        losses.append(loss)
        epoch.append(i)

    import matplotlib.pyplot as plt

    plt.plot(epoch, losses)
    plt.yscale("log")
    plt.show()


if __name__ == "__main__":
    main()
