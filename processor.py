import numpy as np


# data processing, including data imputation, normalization, outlier filtering, feature augmentation and selection
class Processor(object):
    def __init__(self, outlier_factor=10, poly_factor=4, pmode="train"):
        super(Processor, self).__init__()
        self.outlier_factor = outlier_factor
        self.poly_factor = poly_factor
        self.mode = pmode

        # For value of X, y
        self.convert_category_tx = None  # For category column(s)
        self.float_tx = None             # For float columns
        self.x_bias = None               # For linear model bias, add a full-one column
        self.tx = None                   # For X
        self.y = None                    # For y
        self.id = None                   # For id

        self.poly_float_tx = None
        self.poly_tx = None

        # Add descriptions for each column and each row
        self.float_col_names = []
        self.origin_ids = []
        self.category_col_names = None
        self.full_col_names = None
        self.poly_tx = None
        self.poly_float_col_names = None
        self.poly_full_col_names = None

        # For feature selection
        self.select_poly_float_col_names = None
        self.select_full_col_names = None
        self.select_poly_tx = None

    def load_data(self, path="train.csv"):
        with open(path, "r") as f:
            title_list = f.readline().split(",")

        id_col = title_list.index("Id")
        label_col = title_list.index("Prediction")
        jet_num_col = title_list.index("PRI_jet_num")
        special_col = [id_col, label_col, jet_num_col]

        float_col_ids = []
        for idx in range(len(title_list)):
            if idx not in special_col:
                float_col_ids.append(idx)
                self.float_col_names.append(title_list[idx])

        convert_category_y = {"s": 1.0, "b": 0.0, "?": -1.0}
        # one-hot category of jet num
        convert_category_jet_num = {"0": [0.0, 0.0, 0.0, 1.0], "1": [0.0, 0.0, 1.0, 0.0],
                                    "2": [0.0, 1.0, 0.0, 0.0], "3": [1.0, 0.0, 0.0, 0.0]}
        self.category_col_names = ['jetnum3', 'jetnum2', 'jetnum1', 'jetnum0']

        ids = np.genfromtxt(file_path, dtype=str, delimiter=",", skip_header=1, usecols=[id_col])
        self.id = ids

        y_label = np.genfromtxt(file_path, dtype=str, delimiter=",", skip_header=1, usecols=[label_col])
        self.y = np.array(list(map(lambda x: convert_category_y[x], y_label)))

        category_tx = np.genfromtxt(file_path, dtype=str, delimiter=",", skip_header=1, usecols=[jet_num_col])
        self.convert_category_tx = np.array(list(map(lambda x: convert_category_jet_num[x], category_tx)))

        self.float_tx = np.genfromtxt(file_path, dtype=float, delimiter=",", skip_header=1, usecols=float_col_ids)

        self.x_bias = np.ones(len(self.y))

        self.full_col_names = ['x_bias'] + self.float_col_names + self.category_col_names
        self.full_col_names = [x.replace("\n", "") for x in self.full_col_names]
        # self.full_col_names = self.float_col_names + self.category_col_names
        self.tx = np.c_[np.ones(len(self.y)), self.float_tx, self.convert_category_tx]
        # self.tx = np.c_[self.float_tx, self.convert_category_tx]
        self.origin_ids = np.arange(0, self.tx.shape[0])

    def data_imputation(self):
        if not hasattr(self, 'float_tx'):
            raise ValueError("Make sure that you have load data")
        self.float_tx[self.float_tx == -999.0] = 0.0
        self.float_tx[np.isnan(self.float_tx)] = 0.0

    # Make sure to polynomial first then normalize
    def normalize(self):
        if not hasattr(self, 'float_tx'):
            raise ValueError("Make sure that you have load data")

        self.float_tx = (self.float_tx - np.nanmean(self.float_tx, axis=0)) / np.nanstd(self.float_tx, axis=0)
        self.tx = np.c_[np.ones(len(self.y)), self.float_tx, self.convert_category_tx]

        if self.poly_float_tx is not None:
            self.poly_float_tx = (self.poly_float_tx - np.nanmean(self.poly_float_tx, axis=0)) / np.nanstd(self.poly_float_tx, axis=0)
            self.poly_tx = np.c_[np.ones(len(self.y)), self.poly_float_tx, self.convert_category_tx]

    def nan_percentage(self):
        if not hasattr(self, 'float_tx'):
            raise ValueError("Make sure that you have load data")
        self.float_tx[self.float_tx == -999.0] = np.nan
        self.float_tx[np.isnan(self.float_tx)] = np.nan

        for i in range(self.float_tx.shape[1]):
            num_samples = self.float_tx.shape[0]
            nan_ratio = np.count_nonzero(np.isnan(self.float_tx[:, i])) * 1.0 / num_samples
            print("Nan ration of feature {}: \t{}".format(self.float_col_names[i], nan_ratio))

    # filter outlier over <mean - outlier_factor * std, mean + outlier_factor * std>
    def filter_outliers(self, outlier_factor=None):
        if not hasattr(self, 'float_tx'):
            raise ValueError("Make sure that you have load data")
        if self.mode != "test":  # do not filter outliers for test set
            factor = outlier_factor if outlier_factor else self.outlier_factor
            for i in range(self.float_tx.shape[1]):
                filter_flag = abs(self.float_tx[:, i] - np.mean(self.float_tx[:, i])) < factor * np.std(self.float_tx[:, i])
                self.float_tx = self.float_tx[filter_flag]
                self.convert_category_tx = self.convert_category_tx[filter_flag]
                self.x_bias = self.x_bias[filter_flag]
                self.tx = self.tx[filter_flag]
                self.y = self.y[filter_flag]
                self.origin_ids = self.origin_ids[filter_flag]
                assert self.origin_ids.shape[0] == self.tx.shape[0]
                assert self.origin_ids.shape[0] == self.y.shape[0]
            assert len(self.full_col_names) == self.tx.shape[1]

    # For polynomial feature augmentation
    def polynomial(self, poly_factor=None):
        assert len(self.float_col_names) == self.float_tx.shape[1]
        if not hasattr(self, 'float_tx'):
            raise ValueError("Make sure that you have load data")
        poly_factor = poly_factor if poly_factor else self.poly_factor
        poly_float_tx = []
        poly_float_names = []
        for i in range(len(self.float_col_names)):
            col_name = self.float_col_names[i]
            col = self.float_tx[:, i]

            poly_columns = [col**j for j in range(1, poly_factor + 1)]
            poly_col_names = [col_name.replace("\n", "") + "_" + str(j) for j in range(1, poly_factor + 1)]

            poly_float_tx += poly_columns
            poly_float_names += poly_col_names

        self.poly_float_tx = np.stack(poly_float_tx)
        self.poly_float_tx = self.poly_float_tx.T
        self.poly_float_col_names = poly_float_names

        self.poly_tx = np.c_[np.ones(len(self.y)), self.poly_float_tx, self.convert_category_tx]
        self.poly_full_col_names = ['x_bias'] + self.poly_float_col_names + self.category_col_names

    def select_features(self, select_features_poly1, poly_factor=None):
        select_features_poly1.remove('x_bias')
        self.select_poly_float_col_names = []
        poly_factor = poly_factor if poly_factor else self.poly_factor

        # From poly=1 to poly=poly_factor
        for feature_name in select_features_poly1:
            self.select_poly_float_col_names += [feature_name.replace("_1", "_" + str(i)) for i in range(1, poly_factor+1)]

        self.select_full_col_names = ['x_bias'] + self.select_poly_float_col_names + self.category_col_names
        select_full_col_name_indices = [self.poly_full_col_names.index(feature) for feature in self.select_full_col_names]
        self.select_poly_tx = self.poly_tx[:, select_full_col_name_indices]


if __name__ == "__main__":
    processor_train = Processor(outlier_factor=10, poly_factor=4, pmode="train")
    processor_test = Processor(outlier_factor=-1, poly_factor=4, pmode="test")
    for processor in [processor_train, processor_test]:
        if processor.mode == "test":
            save_id = True
            save_y = False
            file_path = "./test.csv"
        else:
            save_id = False
            save_y = True
            file_path = "./train.csv"
        save_original_tx = True
        save_poly_tx = True
        save_select_tx = True

        processor.load_data(path=file_path)
        processor.nan_percentage()

        print("The original number of data: {}".format(processor.tx.shape[0]))
        processor.data_imputation()
        processor.filter_outliers()
        print("The final number of data after pre-processing: {}".format(processor.tx.shape[0]))
        print("Full col names: {}".format(processor.full_col_names))
        processor.polynomial()
        processor.normalize()

        if save_id:
            print("Preprocessor id: {}".format(processor.id))
            np.savetxt("test_id.csv", np.array(processor.id), delimiter=",", header="Id", comments="", fmt='%s')

        # Save y
        if save_y:
            file_name = "filter_factor_" + str(processor.outlier_factor) + "_y.csv"
            np.savetxt(file_name, processor.y, delimiter=",", header="y_label", comments="")

        # Save original tx
        if save_original_tx:
            assert processor.tx.shape[0] == processor.y.shape[0]
            if processor.mode == "test":
                file_name = "test_tx.csv"
            else:
                file_name = "filter_factor_" + str(processor.outlier_factor) + "_tx.csv"
            print("Len of column names: {}".format(len(processor.full_col_names)))
            header_str = ",".join(processor.full_col_names)
            np.savetxt(file_name, processor.tx, delimiter=",", header=header_str, comments="")
            # with open("column_file.csv", 'w', newline='') as f:
            #     f.write(",".join(preprocessor.full_col_names))

        # Save poly_tx
        if save_poly_tx:
            assert processor.poly_tx.shape[0] == processor.y.shape[0]
            assert processor.poly_tx.shape[1] == len(processor.poly_full_col_names)
            if processor.mode == "test":
                file_name = "test_poly_" + str(processor.poly_factor) + "_tx.csv"
            else:
                file_name = "filter_factor_" + str(processor.outlier_factor) + "_poly_" \
                            + str(processor.poly_factor) + "_tx.csv"

            poly_header_str = ",".join(processor.poly_full_col_names)
            np.savetxt(file_name, processor.poly_tx, delimiter=",", header=poly_header_str, comments="")
            print("poly_full_column_names: {}".format(processor.poly_full_col_names))
            # with open("column_file_poly.csv", "w") as f:
            #     f.write(",".join(preprocessor.poly_full_col_names))

        # Following is for feature selection
        if save_select_tx:
            # poly_factor = 4, SGD with fixed learning rate, best performance
            select_features = ['DER_deltar_tau_lep_1', 'DER_mass_vis_1', 'DER_mass_MMC_1', 'DER_pt_ratio_lep_tau_1',
                               'x_bias', 'PRI_jet_leading_pt_1', 'PRI_lep_pt_1', 'DER_deltaeta_jet_jet_1',
                               'DER_mass_jet_jet_1', 'PRI_tau_pt_1', 'DER_sum_pt_1', 'PRI_jet_subleading_pt_1',
                               'DER_mass_transverse_met_lep_1', 'DER_prodeta_jet_jet_1', 'PRI_met_1', 'PRI_jet_all_pt_1',
                               'DER_lep_eta_centrality_1', 'DER_pt_h_1', 'PRI_met_sumet_1', 'DER_met_phi_centrality_1']
            processor.select_features(select_features_poly1=select_features)
            print("select_poly_tx.shape: {}".format(processor.select_poly_tx.shape))
            assert processor.select_poly_tx.shape[0] == processor.y.shape[0]
            assert processor.select_poly_tx.shape[1] == len(processor.select_full_col_names)
            if processor.mode == "test":
                # As what described in the report, we don't apply feature selection on our final model as
                # feature selection even harms model's performance
                print("We don't apply feature selection on our final model since feature selection harms performance!")
            else:
                file_name = "select_feature_top20_" + "filter_factor_" + str(processor.outlier_factor) + \
                            "_poly_" + str(processor.poly_factor) + "_tx.csv"
                poly_select_header_str = ",".join(processor.select_full_col_names)
                np.savetxt(file_name, processor.select_poly_tx, delimiter=",", header=poly_select_header_str, comments="")
