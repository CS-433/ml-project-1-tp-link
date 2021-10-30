from implementations import *
from toolkits import *


# cross validation, return average train/validation loss, accuracy, precision, recall, F1-measure and weights
def cross_validation(y, tx, lambda_=None, gamma=None, init_w=None, max_epoch_iters=1000, k_cross=10,
                     model="reg_logistic_dynamic", batch_size=1, dynamic_lr=True,
                     half_lr_count=2, early_stop_count=4):
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
        if model == "reg_logistic_dynamic":
            weights[k], loss = reg_logistic_dynamic(y_train, tx_train, y_valid, tx_valid, initial_w=weights[k],
                                                    max_epoch_iters=max_epoch_iters, gamma=gamma, batch_size=batch_size,
                                                    lambda_=lambda_, dynamic_lr=dynamic_lr, k_cross=k_cross,
                                                    half_lr_count=half_lr_count, early_stop_count=early_stop_count)
            y_predict_train, tr_loss = predict_binary(y_train, tx_train, weights[k], loss_type="logistic")
            y_predict, va_loss = predict_binary(y_valid, tx_valid, weights[k], loss_type="logistic")
        else:
            raise ValueError("model must be reg_logistic_dynamic")

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


# train with different hyper-parameters, using cross validation and write results in csv files, return k sets of weights
def train_valid(y, tx, lambdas=None, gammas=None, max_epoch_iters=None, init_w=None, k_cross=10,
                model="reg_logistic_dynamic", batches=None, dynamic_lr=False, poly_factor=4, half_lr_count=2,
                early_stop_count=4):
    with open("results_" + model + "_k_" + str(k_cross) + "_poly_" + str(poly_factor) + ".csv", "w") as f:
        if model == "reg_logistic_dynamic":
            for lambda_ in lambdas:
                for gamma in gammas:
                    for batch_size in batches:
                        weights = [init_w for _ in range(k_cross)]
                        weights, results = cross_validation(y, tx, gamma=gamma, max_epoch_iters=max_epoch_iters,
                                                            init_w=weights, batch_size=batch_size,
                                                            dynamic_lr=dynamic_lr, half_lr_count=half_lr_count,
                                                            early_stop_count=early_stop_count,
                                                            k_cross=k_cross, lambda_=lambda_, model=model)
                        f.write("lambda_" + str(lambda_) + "_gamma_" + str(gamma) + "_batch_" + str(batch_size) + "\n")
                        write_results_valid(f, results, weights, tx.shape[1], k_cross)
    return weights


# make predictions on test set using models in cross validation, based on vote scheme, write results in csv file
def predict_test(tx_test, ids_test, weights, poly_factor=4, k_cross=10, model="reg_logistic_dynamic"):
    with open("results_" + model + "_poly_" + str(poly_factor) + ".csv", "w") as f:
        if model == "reg_logistic_dynamic":
            y_predicts_test = []  # predictions of the k models in cross validation
            for weight in weights:
                y_predicts_test.append(predict_binary_test(tx_test, weight, model_type="logistic").copy())
            # vote for final prediction, follow the majority
            y_predicts_test_final = \
                list(map(lambda x: 1 if x > k_cross//2 else -1, np.sum(np.array(y_predicts_test), axis=0)))
        else:
            raise ValueError("model must be reg_logistic_dynamic")
    write_results_test(f, ids_test, y_predicts_test_final)


if __name__ == '__main__':
    outlier_factor = 10  # factor for outlier filtering
    poly_factors = [4]  # from [1, 2, 3, 4, 5, 6]
    select_feature = False  # whether conduct feature selection or not
    for poly in poly_factors:
        # load processed data, given by processor.py
        labels, features, features_test, ids_test_list = \
            load_data_processed(file_path="./", outlier_factor=outlier_factor, poly_factor=poly,
                                select_feature=select_feature)
        random_seed = 13
        cross_valid = 10  # 10-fold cross validation
        hyper_lambda = [1e-5]  # search from [1e-6, 1e-5, 1e-4, 1e-3, 0.01]
        hyper_gamma = [2e-3]  # search from [5e-4, 1e-3, 2e-3, 5e-3, 0.01]
        hyper_batch = [1]  # still use SGD with batch_size=1, could be changed to mini-batch to improve efficiency
        dynamic = True  # whether use dynamic learning rate or not
        half_lr = 2  # halve learning rate if validation loss does not decrease for 2 consecutive epochs
        early_stop = 4  # stop training if validation loss does not decrease for 4 consecutive epochs
        max_epochs = 1000  # maximum training epochs

        np.random.seed(random_seed)

        # initialize weights by random sampling from uniform distribution
        initial_w = np.random.uniform(low=-2.0, high=2.0, size=features.shape[1])  # np.random.randn(features.shape[1])

        # shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(labels)))
        shuffled_y = labels[shuffle_indices]
        shuffled_tx = features[shuffle_indices]

        info_str = "Training Regularized Logistic Regression, Outlier Filtering Factor-" + str(outlier_factor) \
                   + ", Feature Augmentation Polynomial Factor-" + str(poly)
        if select_feature:
            info_str += ", Feature Selection On"
        if dynamic:
            info_str += ", Dynamic Learning Rate On"

        print(info_str + ":")
        # do training based on k-fold cross validation
        k_weights = train_valid(shuffled_y, shuffled_tx, gammas=hyper_gamma, max_epoch_iters=max_epochs,
                                init_w=initial_w.copy(), k_cross=cross_valid, model="reg_logistic_dynamic",
                                batches=hyper_batch, dynamic_lr=dynamic, lambdas=hyper_lambda, poly_factor=poly,
                                half_lr_count=half_lr, early_stop_count=early_stop)

        # use the k models from cross validation to jointly do the prediction on test set (via vote scheme)
        predict_test(features_test, ids_test_list, k_weights, poly_factor=poly, k_cross=cross_valid,
                     model="reg_logistic_dynamic")
