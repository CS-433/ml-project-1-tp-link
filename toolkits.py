import numpy as np


# load data processed by processor.py, used for our improved model
def load_data_processed(file_path="./", outlier_factor=10, poly_factor=4, select_feature=False):
    path_tx = "filter_factor_" + str(outlier_factor) + "_poly_" + str(poly_factor) + "_tx.csv"
    if select_feature:
        path_tx = "select_feature_top20_" + path_tx
    path_tx = file_path + path_tx
    tx = np.genfromtxt(path_tx, dtype=float, delimiter=",", skip_header=1)
    path_y = file_path + "filter_factor_" + str(outlier_factor) + "_y.csv"
    y = np.genfromtxt(path_y, dtype=float, delimiter=",", skip_header=1)
    path_tx_test = file_path + "test_poly_" + str(poly_factor) + "_tx.csv"
    tx_test = np.genfromtxt(path_tx_test, dtype=float, delimiter=",", skip_header=1)
    id_test_path = file_path + "test_id.csv"
    id_test = np.genfromtxt(id_test_path, dtype=float, delimiter=",", skip_header=1)
    return y, tx, tx_test, id_test


# load original data with only data imputation and normalization
def load_data_origin(file_path="./", normalize=True):

    with open(file_path, "r") as f:
        title_list = f.readline().split(",")

    id_col = title_list.index("Id")
    label_col = title_list.index("Prediction")
    jet_num_col = title_list.index("PRI_jet_num")
    special_col = [id_col, label_col, jet_num_col]

    float_col = []
    for idx in range(len(title_list)):
        if idx not in special_col:
            float_col.append(idx)

    convert_category_y = {"s": 1.0, "b": 0.0, "?": -1.0}
    convert_category_jet_num = {"0": [0.0, 0.0, 0.0, 1.0], "1": [0.0, 0.0, 1.0, 0.0],
                                "2": [0.0, 1.0, 0.0, 0.0], "3": [1.0, 0.0, 0.0, 0.0]}

    ids = np.genfromtxt(file_path, dtype=str, delimiter=",", skip_header=1, usecols=[id_col])

    y_label = np.genfromtxt(file_path, dtype=str, delimiter=",", skip_header=1, usecols=[label_col])
    y = np.array(list(map(lambda x: convert_category_y[x], y_label)))

    category_tx = np.genfromtxt(file_path, dtype=str, delimiter=",", skip_header=1, usecols=[jet_num_col])
    convert_category_tx = np.array(list(map(lambda x: convert_category_jet_num[x], category_tx)))

    float_tx = np.genfromtxt(file_path, dtype=float, delimiter=",", skip_header=1, usecols=float_col)

    float_tx[float_tx == -999] = 0.0
    if normalize:
        float_tx = (float_tx - np.mean(float_tx, axis=0)) / np.std(float_tx, axis=0)

    tx = np.c_[np.ones(len(y)), float_tx, convert_category_tx]

    return ids, y, tx


# training data batch generator
def batch_iter(y, tx, batch_size, num_batches=1, seed=7, shuffle=True):
    """
    :param y: labels, size: (N,)
    :param tx: features, size: (N,D)
    :param batch_size: size of mini_batch B
    :param num_batches: number of batches generated K
    :param seed: random seed for shuffle
    :param shuffle: if shuffle the data
    :return: mini-batch data: ((y_mini_batch_1, tx_mini_batch_1), ..., (y_mini_batch_K, tx_mini_batch_K))
    """
    data_size = y.shape[0]
    np.random.seed(seed)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx

    for batch_num in range(int(num_batches)):
        start_index = int(batch_num * batch_size) % data_size
        end_index = int((batch_num + 1) * batch_size) % data_size
        if start_index < end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
        else:
            if shuffle:
                new_shuffle_indices = np.random.permutation(np.arange(data_size))
                new_shuffled_y = y[new_shuffle_indices]
                new_shuffled_tx = tx[new_shuffle_indices]
            else:
                new_shuffled_y = y
                new_shuffled_tx = tx
            yield np.r_[shuffled_y[start_index:], new_shuffled_y[:end_index]], np.r_[shuffled_tx[start_index:],
                                                                                     new_shuffled_tx[:end_index]]
            shuffled_y = new_shuffled_y
            shuffled_tx = new_shuffled_tx


# make prediction and calculate loss on train/validation set
def predict_binary(y, tx, w, loss_type="logistic"):
    """
    :param y: ground truth labels, size: (N_Test,)
    :param tx: input features, size: (N_Test,D)
    :param w: trained weights
    :param loss_type: model type: ["mse", "rmse", "logistic"]
    :return: predict_y, test_loss
    """
    z = np.dot(tx, w)
    if loss_type == "mse":
        test_loss = 0.5 * np.mean((y - z)**2)
    elif loss_type == "rmse":
        test_loss = np.sqrt(np.mean((y - z)**2))
    elif loss_type == "logistic":
        test_loss = np.mean(np.log(1 + np.exp(z)) - y * z)
    else:
        raise ValueError("loss_type must be mse, rmse or logistic")

    if loss_type == "logistic":
        p_pred = np.exp(z) / (1 + np.exp(z))
    else:
        p_pred = z

    predict_y = list(map(lambda x: 0 if x < 0.5 else 1, p_pred))

    return predict_y, test_loss


# make prediction on test set
def predict_binary_test(tx, w, model_type="logistic", mode="test"):
    """
    :param tx: input features, size: (N_Test,D)
    :param w: trained weights
    :param model_type: model type: ["mse", "rmse", "logistic"]
    :param mode: prediction mode
    :return: predict_y
    """
    z = np.dot(tx, w)

    if model_type == "logistic":
        p_pred = np.exp(z) / (1 + np.exp(z))
    else:
        p_pred = z

    if mode == "train":  # original labels we predict are in {0, 1}
        predict_y = list(map(lambda x: 0 if x < 0.5 else 1, p_pred))
    elif mode == "test":  # predictions submitted to platform are in {-1, 1}
        predict_y = list(map(lambda x: -1 if x < 0.5 else 1, p_pred))
    else:
        predict_y = None

    return predict_y


# compute accuracy precision, recall and F1-measure of binary classification
def compute_prf_binary(label_y, predict_y):
    """
    :param label_y: ground truth labels
    :param predict_y: model predictions
    :return: precision, recall, f1
    """
    tp, fp, tn, fn = 1e-5, 1e-5, 1e-5, 1e-5
    for idx, label in enumerate(label_y):
        if label == 1:
            if predict_y[idx] == 1:
                tp += 1
            else:
                fn += 1
        else:
            if predict_y[idx] == 1:
                fp += 1
            else:
                tn += 1
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return accuracy, precision, recall, f1


# write validation results, including scores of metrics and average output weights of cross validation
def write_results_valid(file_path, results, weights, feature_dim, k_cross=10):
    title = ["train_loss", "valid_loss", "tr_accuracy", "tr_precision", "tr_recall", "tr_f1_measure",
             "va_accuracy", "va_precision", "va_recall", "va_f1_measure"]
    with open(file_path, "a") as f:
        f.write(",".join(title) + "\n")
        f.write(",".join([str(x) for x in results]) + "\n")
        weight_avg = np.zeros(feature_dim)
        for idx in range(k_cross):
            weight_avg = weight_avg + weights[idx]
        weight_avg = weight_avg / k_cross
        f.write("weight_avg" + "\n")
        f.write(",".join([str(x) for x in list(weight_avg)]) + "\n")


# write test results, including ids and predictions
def write_results_test(file_path, ids, y_predicts):
    title = ["Id", "Prediction"]
    with open(file_path, "w") as f:
        f.write(",".join(title) + "\n")
        for test_id, y_predict in zip(ids, y_predicts):
            f.write(",".join([str(test_id), str(y_predict)]) + "\n")
