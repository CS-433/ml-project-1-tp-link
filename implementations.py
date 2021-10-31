import numpy as np
from toolkits import batch_iter, predict_binary


# linear regression using gradient descent
def least_squares_GD(y, tx, initial_w=None, max_iters=100, gamma=0.25):
    """
    :param y: labels, size: (N,)
    :param tx: features, size: (N,D)
    :param initial_w: initial weight, size: (D,)
    :param max_iters: number of steps to run
    :param gamma: step size for (stochastic) gradient descent
    :return: (w, loss): last weight vector and loss, size: ((D,), 1)
    """
    N = y.shape[0]
    losses = []
    w = initial_w if initial_w is not None else np.zeros(tx.shape[1])
    for mini_y, mini_tx in batch_iter(y, tx, batch_size=N, num_batches=max_iters, shuffle=True):
        loss = 0.5 * np.mean((mini_y - np.dot(mini_tx, w))**2)
        e = mini_y - np.dot(mini_tx, w)
        gradient = - np.dot(mini_tx.T, e) / N
        w = w - (gamma*gradient)
        losses.append(loss)
    return w, losses[-1]


# linear regression using stochastic gradient descent
def least_squares_SGD(y, tx, initial_w=None, max_iters=450000, gamma=1e-3):
    """
    :param y: labels, size: (N,)
    :param tx: features, size: (N,D)
    :param initial_w: initial weight, size: (D,)
    :param max_iters: number of steps to run
    :param gamma: step size for (stochastic) gradient descent
    :return: (w, loss): last weight vector and loss, size: ((D,), 1)
    """
    losses = []
    w = initial_w if initial_w is not None else np.zeros(tx.shape[1])
    for mini_y, mini_tx in batch_iter(y, tx, batch_size=1, num_batches=max_iters, shuffle=True):
        loss = 0.5 * np.mean((mini_y - np.dot(mini_tx, w))**2)
        e = mini_y - np.dot(mini_tx, w)
        gradient = - np.dot(mini_tx.T, e)
        w = w - (gamma*gradient)
        losses.append(loss)
    return w, losses[-1]


# least squares regression using normal equations
def least_squares(y, tx):
    """
    :param y: labels, size: (N,)
    :param tx: features, size: (N,D)
    :return: (w, loss): last weight vector and loss, size: ((D,), 1)
    """
    w = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    loss = 0.5 * np.mean((y - np.dot(tx, w))**2)

    return w, loss


# ridge regression using normal equations
def ridge_regression(y, tx, lambda_=1e-5):
    """
    :param y: labels, size: (N,)
    :param tx: features, size: (N,D)
    :param lambda_: regularization coefficient
    :return: (w, loss): last weight vector and loss, size: ((D,), 1)
    """
    N = y.shape[0]
    D = tx.shape[1]
    w = np.linalg.solve(np.dot(tx.T, tx) + 2 * N * lambda_ * np.identity(D), np.dot(tx.T, y))
    loss = 0.5 * np.mean((y - np.dot(tx, w))**2)
    return w, loss


# logistic regression using SGD
def logistic_regression(y, tx, initial_w=None, max_iters=225000, gamma=2e-3):
    """
    :param y: labels, size: (N,)
    :param tx: features, size: (N,D)
    :param initial_w: initial weight, size: (D,)
    :param max_iters: number of steps to run
    :param gamma: step size for (stochastic) gradient descent
    :return: (w, loss): last weight vector and loss, size: ((D,), 1)
    """
    losses = []
    w = initial_w if initial_w is not None else np.zeros(tx.shape[1])
    for mini_y, mini_tx in batch_iter(y, tx, batch_size=1, num_batches=max_iters, shuffle=True):
        z = np.dot(mini_tx, w)
        loss = np.mean(np.log(1 + np.exp(z)) - mini_y * z)
        gradient = np.dot(mini_tx.T, (np.exp(z) / (1 + np.exp(z)) - mini_y))
        w = w - (gamma*gradient)
        losses.append(loss)
    return w, losses[-1]


# regularized logistic regression using SGD
def reg_logistic_regression(y, tx, initial_w=None, lambda_=1e-5, max_iters=225000, gamma=2e-3):
    """
    :param y: labels, size: (N,)
    :param tx: features, size: (N,D)
    :param lambda_: regularization coefficient
    :param initial_w: initial weight, size: (D,)
    :param max_iters: number of steps to run
    :param gamma: step size for (stochastic) gradient descent
    :return: (w, loss): last weight vector and loss, size: ((D,), 1)
    """
    losses = []
    w = initial_w if initial_w is not None else np.zeros(tx.shape[1])
    for mini_y, mini_tx in batch_iter(y, tx, batch_size=1, num_batches=max_iters, shuffle=True):
        z = np.dot(mini_tx, w)
        loss = np.mean(np.log(1 + np.exp(z)) - mini_y * z)
        gradient = np.dot(mini_tx.T, (np.exp(z) / (1 + np.exp(z)) - mini_y)) + lambda_ * w
        w = w - (gamma * gradient)
        losses.append(loss)
    return w, losses[-1]


# our improved model, regularized logistic regression with dynamic learning rate,
# combined with outlier filtering and feature augmentation
def reg_logistic_dynamic(y, tx, y_valid, tx_valid, initial_w=None, max_epoch_iters=1000, gamma=2e-3, batch_size=1,
                         lambda_=1e-5, dynamic_lr=True, k_cross=10, half_lr_count=2, early_stop_count=4):
    """
    :param y: labels, size: (N,)
    :param tx: features, size: (N,D)
    :param y_valid: validation set labels, size: (Nv,)
    :param tx_valid: validation set features, size: (Nv,D)
    :param initial_w: initial weight, size: (D,)
    :param max_epoch_iters: number of epochs to run
    :param gamma: step size (related to learning rate) for mini_batch gradient descent
    :param batch_size: batch size for mini_batch gradient descent
    :param lambda_: regularization coefficient
    :param dynamic_lr: if use dynamic learning rate, otherwise fixed
    :param k_cross: number of cross validation fold
    :param half_lr_count: half the learning rate if validation loss increase in count of consecutive epochs
    :param early_stop_count: stop training if validation loss increase in count of consecutive epochs
    :return: (w, loss): last weight vector and loss, size: ((D,), 1)
    """
    data_size = len(y)
    train_valid_ratio = (k_cross - 1) / k_cross
    iters_per_epoch = int(data_size * train_valid_ratio / batch_size)
    lr = gamma
    loss = None
    w = initial_w if initial_w is not None else np.zeros(tx.shape[1])

    if dynamic_lr:
        w_best = w.copy()
        _, min_valid_loss = predict_binary(y_valid, tx_valid, w, loss_type="logistic")
        half_count = half_lr_count
        stop_count = early_stop_count
    else:
        w_best, min_valid_loss, half_count, stop_count = None, None, None, None

    for epoch in range(max_epoch_iters):
        # perform SGD
        for mini_y, mini_tx in batch_iter(y, tx, batch_size=batch_size, num_batches=iters_per_epoch, shuffle=True):
            z = np.dot(mini_tx, w)
            loss = np.mean(np.log(1 + np.exp(z)) - mini_y * z)
            gradient = np.dot(mini_tx.T, (np.exp(z) / (1 + np.exp(z)) - mini_y)) + lambda_ * w
            w = w - (lr * gradient)
        print("Training Epoch: " + str(epoch) + ", Training Loss (MSE): " + str(loss))
        # learning rate control
        if dynamic_lr:
            _, valid_loss = predict_binary(y_valid, tx_valid, w, loss_type="logistic")
            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                w_best = w.copy()
                half_count = half_lr_count
                stop_count = early_stop_count
            else:
                half_count -= 1
                stop_count -= 1
            if half_count == 0:
                lr = lr / 2.0
                half_count = half_lr_count
            if stop_count == 0:
                break

    if dynamic_lr:
        return w_best, loss
    else:
        return w, loss
