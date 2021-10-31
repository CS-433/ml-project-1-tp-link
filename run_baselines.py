from implementations import *
from toolkits import *


# cross validation, return average train/validation loss, accuracy, precision, recall, F1-measure and weights
def cross_validation(y, tx, lambda_=None, gamma=None, iters=None, init_w=None, k_cross=10, model="reg_logistic"):
    data_size = len(y)
    train_loss, valid_loss = [], []
    tr_accuracy, tr_precision, tr_recall, tr_f1_measure = [], [], [], []
    va_accuracy, va_precision, va_recall, va_f1_measure = [], [], [], []
    if init_w is not None:
        weights = init_w
    else:
        weights = [np.zeros(tx.shape[1]) for _ in range(k_cross)]

    for k in range(k_cross):
        start_index = k * data_size // k_cross
        end_index = (k + 1) * data_size // k_cross
        y_valid = y[start_index:end_index]
        tx_valid = tx[start_index:end_index]
        y_train = np.r_[y[:start_index], y[end_index:]]
        tx_train = np.r_[tx[:start_index], tx[end_index:]]

        if model == "least_squares_GD":
            weights[k], loss = least_squares_GD(y_train, tx_train, initial_w=weights[k], max_iters=iters, gamma=gamma)
            y_predict_train, tr_loss = predict_binary(y_train, tx_train, weights[k], loss_type="rmse")
            y_predict, va_loss = predict_binary(y_valid, tx_valid, weights[k], loss_type="rmse")
        elif model == "least_squares_SGD":
            weights[k], loss = least_squares_SGD(y_train, tx_train, initial_w=weights[k], max_iters=iters, gamma=gamma)
            y_predict_train, tr_loss = predict_binary(y_train, tx_train, weights[k], loss_type="rmse")
            y_predict, va_loss = predict_binary(y_valid, tx_valid, weights[k], loss_type="rmse")
        elif model == "least_squares":
            weights[k], loss = least_squares(y_train, tx_train)
            y_predict_train, tr_loss = predict_binary(y_train, tx_train, weights[k], loss_type="rmse")
            y_predict, va_loss = predict_binary(y_valid, tx_valid, weights[k], loss_type="rmse")
        elif model == "ridge":
            weights[k], loss = ridge_regression(y_train, tx_train, lambda_=lambda_)
            y_predict_train, tr_loss = predict_binary(y_train, tx_train, weights[k], loss_type="rmse")
            y_predict, va_loss = predict_binary(y_valid, tx_valid, weights[k], loss_type="rmse")
        elif model == "logistic":
            weights[k], loss = logistic_regression(y_train, tx_train, initial_w=weights[k],
                                                   max_iters=iters, gamma=gamma)
            y_predict_train, tr_loss = predict_binary(y_train, tx_train, weights[k], loss_type="logistic")
            y_predict, va_loss = predict_binary(y_valid, tx_valid, weights[k], loss_type="logistic")
        elif model == "reg_logistic":
            weights[k], loss = reg_logistic_regression(y_train, tx_train, lambda_=lambda_, initial_w=weights[k],
                                                       max_iters=iters, gamma=gamma)
            y_predict_train, tr_loss = predict_binary(y_train, tx_train, weights[k], loss_type="logistic")
            y_predict, va_loss = predict_binary(y_valid, tx_valid, weights[k], loss_type="logistic")
        else:
            raise ValueError("model must be least_squares_GD, least_squares_SGD, least_squares, ridge, " +
                             "logistic or reg_logistic")

        train_loss.append(tr_loss)
        valid_loss.append(va_loss)
        tr_acc, tr_p, tr_r, tr_f1 = compute_prf_binary(y_train, y_predict_train)
        va_acc, va_p, va_r, va_f1 = compute_prf_binary(y_valid, y_predict)
        tr_accuracy.append(tr_acc)
        tr_precision.append(tr_p)
        tr_recall.append(tr_r)
        tr_f1_measure.append(tr_f1)
        va_accuracy.append(va_acc)
        va_precision.append(va_p)
        va_recall.append(va_r)
        va_f1_measure.append(va_f1)

    results = [np.mean(train_loss), np.mean(valid_loss),
               np.mean(tr_accuracy), np.mean(tr_precision), np.mean(tr_recall), np.mean(tr_f1_measure),
               np.mean(va_accuracy), np.mean(va_precision), np.mean(va_recall), np.mean(va_f1_measure)]
    return weights, results


# train with different hyper-parameters, using cross validation and write results in csv files
def train_valid(y, tx, lambdas=None, gammas=None, epoch_iters=None, init_w=None, k_cross=10, model="reg_logistic"):
    train_valid_ratio = (k_cross - 1) / k_cross
    data_size = len(y)
    file_path = "./results_" + model + "_k_" + str(k_cross) + ".csv"
    if model == "least_squares_GD":
        step_iters = epoch_iters
        for gamma in gammas:
            prev_iter = 0
            weights = [init_w for _ in range(k_cross)]
            for step in step_iters:
                weights, results = cross_validation(y, tx, gamma=gamma, iters=step - prev_iter,
                                                    init_w=weights, k_cross=k_cross, model=model)
                prev_iter = step
                print("Training Step: " + str(step))
                with open(file_path, "a") as f:
                    f.write("gamma_" + str(gamma) + "_iters_" + str(step) + "\n")
                write_results_valid(file_path, results, weights, tx.shape[1], k_cross)
    elif model == "least_squares_SGD":
        step_iters = [int(epoch * data_size * train_valid_ratio) for epoch in epoch_iters]
        for gamma in gammas:
            prev_iter = 0
            weights = [init_w for _ in range(k_cross)]
            for step in step_iters:
                weights, results = cross_validation(y, tx, gamma=gamma, iters=step - prev_iter,
                                                    init_w=weights, k_cross=k_cross, model=model)
                prev_iter = step
                print("Training Step: " + str(step))
                with open(file_path, "a") as f:
                    f.write("gamma_" + str(gamma) + "_iters_" + str(step) + "\n")
                write_results_valid(file_path, results, weights, tx.shape[1], k_cross)
    elif model == "least_squares":
        weights, results = cross_validation(y, tx, k_cross=k_cross, model=model)
        write_results_valid(file_path, results, weights, tx.shape[1], k_cross)
    elif model == "ridge":
        for lambda_ in lambdas:
            weights, results = cross_validation(y, tx, lambda_=lambda_, k_cross=k_cross, model=model)
            with open(file_path, "a") as f:
                f.write("lambda_" + str(lambda_) + "\n")
            write_results_valid(file_path, results, weights, tx.shape[1], k_cross)
    elif model == "logistic":
        step_iters = [int(epoch * data_size * train_valid_ratio) for epoch in epoch_iters]
        for gamma in gammas:
            prev_iter = 0
            weights = [init_w for _ in range(k_cross)]
            for step in step_iters:
                weights, results = cross_validation(y, tx, gamma=gamma, iters=step - prev_iter,
                                                        init_w=weights, k_cross=k_cross, model=model)
                prev_iter = step
                print("Training Step: " + str(step))
                with open(file_path, "a") as f:
                    f.write("gamma_" + str(gamma) + "_iters_" + str(step) + "\n")
                write_results_valid(file_path, results, weights, tx.shape[1], k_cross)
    elif model == "reg_logistic":
        step_iters = [int(epoch * data_size * train_valid_ratio) for epoch in epoch_iters]
        for lambda_ in lambdas:
            for gamma in gammas:
                prev_iter = 0
                weights = [init_w for _ in range(k_cross)]
                for step in step_iters:
                    weights, results = cross_validation(y, tx, lambda_=lambda_, gamma=gamma, iters=step - prev_iter,
                                                        init_w=weights, k_cross=k_cross, model=model)
                    prev_iter = step
                    print("Training Step: " + str(step))
                    with open(file_path, "a") as f:
                        f.write("lambda_" + str(lambda_) + "_gamma_" + str(gamma) + "_iters_" + str(step) + "\n")
                    write_results_valid(file_path, results, weights, tx.shape[1], k_cross)


# train on full training set and write predictions on test set in csv file
def train_test(y, tx, tx_test, ids_test, lambda_=None, gamma=None, epochs=None, init_w=None, model="ridge"):
    data_size = len(y)
    file_path = "./results_" + model + ".csv"
    if model == "least_squares_GD":
        iters = epochs
        print("Training on Full Dataset:")
        weight, loss = least_squares_GD(y, tx, initial_w=init_w, max_iters=iters, gamma=gamma)
        print("Predicting on Test Set:")
        y_predicts_test = predict_binary_test(tx_test, weight, model_type="linear", mode="test")
        write_results_test(file_path, ids_test, y_predicts_test)
    elif model == "least_squares_SGD":
        iters = int(epochs * data_size)
        print("Training on Full Dataset:")
        weight, loss = least_squares_SGD(y, tx, initial_w=init_w, max_iters=iters, gamma=gamma)
        print("Predicting on Test Set:")
        y_predicts_test = predict_binary_test(tx_test, weight, model_type="linear", mode="test")
        write_results_test(file_path, ids_test, y_predicts_test)
    elif model == "least_squares":
        print("Training on Full Dataset:")
        weight, loss = least_squares(y, tx)
        print("Predicting on Test Set:")
        y_predicts_test = predict_binary_test(tx_test, weight, model_type="linear", mode="test")
        write_results_test(file_path, ids_test, y_predicts_test)
    elif model == "ridge":
        print("Training on Full Dataset:")
        weight, loss = ridge_regression(y, tx, lambda_=lambda_)
        print("Predicting on Test Set:")
        y_predicts_test = predict_binary_test(tx_test, weight, model_type="linear", mode="test")
        write_results_test(file_path, ids_test, y_predicts_test)
    elif model == "logistic":
        iters = int(epochs * data_size)
        print("Training on Full Dataset:")
        weight, loss = logistic_regression(y, tx, initial_w=init_w, max_iters=iters, gamma=gamma)
        print("Predicting on Test Set:")
        y_predicts_test = predict_binary_test(tx_test, weight, model_type="logistic", mode="test")
        write_results_test(file_path, ids_test, y_predicts_test)
    elif model == "reg_logistic":
        iters = int(epochs * data_size)
        print("Training on Full Dataset:")
        weight, loss = reg_logistic_regression(y, tx, initial_w=init_w, max_iters=iters, gamma=gamma,
                                               lambda_=lambda_)
        print("Predicting on Test Set:")
        y_predicts_test = predict_binary_test(tx_test, weight, model_type="logistic", mode="test")
        write_results_test(file_path, ids_test, y_predicts_test)
    else:
        raise ValueError("model must be least_squares_GD, least_squares_SGD, least_squares, ridge, " +
                         "logistic or reg_logistic")


if __name__ == '__main__':

    # load original data with only imputation and normalization
    print("Load Data:")
    _, labels, features = load_data_origin("train.csv", normalize=True)
    ids_test_list, _, features_test = load_data_origin("test.csv", normalize=True)

    random_seed = 13
    cross_valid = 10  # 10-fold cross validation
    hyper_lambda = [1e-5]  # search from [1e-6, 1e-5, 1e-4, 1e-3, 0.01]
    hyper_gamma_linear_GD = [0.25]  # search from [0.01, 0.05, 0.1, 0.25, 0.5]
    hyper_gamma_linear_SGD = [1e-3]  # search from [5e-4, 1e-3, 2e-3, 5e-3, 0.01]
    hyper_gamma_logistic_SGD = [2e-3]  # search from [5e-4, 1e-3, 2e-3, 5e-3, 0.01]
    epochs_GD = [x for x in range(1, 101)]  # max epoch 100
    epochs_SGD = [x for x in range(1, 6)]  # # max epoch 5

    np.random.seed(random_seed)

    # initialize weights by random sampling from uniform distribution
    initial_w = np.random.uniform(low=-2.0, high=2.0, size=features.shape[1])

    # shuffle the data
    shuffle_indices = np.random.permutation(np.arange(len(labels)))
    shuffled_y = labels[shuffle_indices]
    shuffled_tx = features[shuffle_indices]

    print("Training least squares Gradient Descent:")
    # do training based on k-fold cross validation
    train_valid(shuffled_y, shuffled_tx, gammas=hyper_gamma_linear_GD, epoch_iters=epochs_GD,
                init_w=initial_w.copy(), k_cross=cross_valid, model="least_squares_GD")

    # do training on full training set and predict on test set
    fgamma = 0.25  # gamma that finally selected
    fepochs = 100  # training epochs that finally selected
    train_test(shuffled_y, shuffled_tx, features_test, ids_test_list, gamma=fgamma, epochs=fepochs,
               init_w=initial_w.copy(), model="least_squares_GD")

    print("Training least squares Stochastic Gradient Descent:")
    # do training based on k-fold cross validation
    train_valid(shuffled_y, shuffled_tx, gammas=hyper_gamma_linear_SGD, epoch_iters=epochs_SGD,
                init_w=initial_w.copy(), k_cross=cross_valid, model="least_squares_SGD")

    # do training on full training set and predict on test set
    fgamma = 1e-3  # gamma that finally selected
    fepochs = 2  # training epochs that finally selected
    train_test(shuffled_y, shuffled_tx, features_test, ids_test_list, gamma=fgamma, epochs=fepochs,
               init_w=initial_w.copy(), model="least_squares_SGD")

    print("Training least squares:")
    # do training based on k-fold cross validation
    train_valid(shuffled_y, shuffled_tx, k_cross=cross_valid, model="least_squares")

    # do training on full training set and predict on test set
    train_test(shuffled_y, shuffled_tx, features_test, ids_test_list, model="least_squares")

    print("Training Ridge Regression:")
    # do training based on k-fold cross validation
    train_valid(shuffled_y, shuffled_tx, lambdas=hyper_lambda, k_cross=cross_valid, model="ridge")

    # do training on full training set and prediction on test set
    flambda_ = 1e-5  # lambda_ that finally selected
    train_test(shuffled_y, shuffled_tx, features_test, ids_test_list, lambda_=flambda_, model="ridge")

    print("Training Logistic Regression:")
    # do training based on k-fold cross validation
    train_valid(shuffled_y, shuffled_tx, gammas=hyper_gamma_logistic_SGD, epoch_iters=epochs_SGD,
                init_w=initial_w.copy(), k_cross=cross_valid, model="logistic")

    # do training on full training set and prediction on test set
    fgamma = 2e-3  # gamma that finally selected
    fepochs = 1  # training epochs that finally selected
    train_test(shuffled_y, shuffled_tx, features_test, ids_test_list, gamma=fgamma, epochs=fepochs,
               init_w=initial_w.copy(), model="logistic")

    print("Training Regularized Logistic Regression:")
    # do training based on k-fold cross validation
    train_valid(shuffled_y, shuffled_tx, lambdas=hyper_lambda, gammas=hyper_gamma_logistic_SGD,
                epoch_iters=epochs_SGD, init_w=initial_w.copy(), k_cross=cross_valid, model="reg_logistic")

    # do training on full training set and prediction on test set
    fgamma = 2e-3  # gamma that finally selected
    fepochs = 1  # training epochs that finally selected
    flambda_ = 1e-5  # lambda_ that finally selected
    train_test(shuffled_y, shuffled_tx, features_test, ids_test_list, gamma=fgamma, epochs=fepochs, lambda_=flambda_,
               init_w=initial_w.copy(), model="reg_logistic")
