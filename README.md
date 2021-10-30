# Machine Learning Fall 2021 Project 1 (Team: TP-Link)
This is our code for the Project 1 of Machine Learning Fall 2021
- Team Name: TP-Link
- Team Member:
1.Silin Gao (silin.gao@epfl.ch)
2.Shaobo Cui (shaobo.cui@epfl.ch)
3.Dongge Wang (dongge.wang@epfl.ch)

## Requirements
- Python
- Numpy
- Matplotlib

## Scripts
- implementations.py: implementations of baseline models and our improved model.
- processor.py: data processor, including data imputation, normalization, outlier filtering, feature augmentation and selection.
- run_baselines.py: script for training and evaluating baseline models, including cross validation and test set prediction.
- run.py: script for training and evaluating our improved model, including cross validation and test set prediction.
- toolkits.py: toolkits including data loader, batch generator, metrics computer, file writer, etc.
- plot_weights.py: script for plotting output weights of features in reg_logistic_regression (outlier factor = 10, polynomial factor = 4), used for feature selection.

## Running Experiments
### Data Processing
```
python processor.py
```
Outputs:
- filter_factor_${outlier_factor}_y.csv: training labels after outlier filtering.
- filter_factor_${outlier_factor}_tx.csv: training features after outlier filtering.
- filter_factor_${outlier_factor}_poly_${polynomial_factor}_tx.csv: training features after outlier filtering and feature augmentation.
- select_feature_top20_filter_factor_${outlier_factor}_poly_${polynomial_factor}_tx.csv: training features after outlier filtering, feature augmentation and feature selection.
- test_id.csv: sample ids in test set.
- test_tx.csv: original test features.
- test_poly_${polynomial_factor}_tx.csv: test features after feature augmentation.


### Baseline Training and Evaluation
```
python run_baselines.py
```
Note: Baseline running directly loads and pre-processes the original data (train.csv and test.csv), which is independent from the data processing.
Outputs:
- results_${model_name}_k_${cross_validation_folds}.csv: cross validation results.
- results_${model_name}.csv: final prediction results on test set.

### Improved Model Training and Evaluation
```
python run.py
```
Outputs:
- results_reg_logistic_dynamic_k_${cross_validation_folds}_poly_${polynomial_factor}.csv: cross validation results.
- results_reg_logistic_dynamic_poly_${polynomial_factor}.csv: final prediction results on test set (best submission).
