# Machine Learning Fall 2021 Project 1 (Team: TP-Link)
This is our code for the Project 1 of Machine Learning Fall 2021
- Team Name: TP-Link
- Team Member:
1. Silin Gao (silin.gao@epfl.ch)
2. Shaobo Cui (shaobo.cui@epfl.ch)
3. Dongge Wang (dongge.wang@epfl.ch)

## Requirements
- Python
- Numpy
- Matplotlib

## Scripts
- ***implementations.py***: implementations of baseline models and our improved model.
- ***processor.py***: data processor, including data imputation, normalization, outlier filtering, feature augmentation and selection.
- ***run_baselines.py***: script for training and evaluating baseline models, including cross validation and test set prediction.
- ***run.py***: script for training and evaluating our improved model, including cross validation and test set prediction.
- ***toolkits.py***: toolkits including data loader, batch generator, metrics computer, file writer, etc.
- ***plot_weights.py***: script for plotting output weights of features in reg_logistic_regression (outlier factor = 10, polynomial factor = 4), used for feature selection.

## Running Experiments
### Data Processing
Place the original training and test sets (***train.csv*** and ***test.csv***) in the root directory of our project.
```
python processor.py
```
Outputs:
- ***filter\_factor\_${outlier_factor}\_y.csv***: training labels after outlier filtering.
- ***filter\_factor\_${outlier_factor}\_tx.csv***: training features after outlier filtering.
- ***filter\_factor\_${outlier_factor}\_poly\_${polynomial_factor}\_tx.csv***: training features after outlier filtering and feature augmentation.
- ***select\_feature_top20_filter_factor\_${outlier_factor}\_poly\_${polynomial_factor}\_tx.csv***: training features after outlier filtering, feature augmentation and feature selection.
- ***test\_id.csv***: sample ids in test set.
- ***test\_tx.csv***: original test features.
- ***test\_poly\_${polynomial_factor}\_tx.csv***: test features after feature augmentation.


### Baseline Training and Evaluation
```
python run_baselines.py
```
Note: Baseline running directly loads and pre-processes the original data (train.csv and test.csv), which is independent from the data processing.
Outputs:
- ***results\_${model_name}\_k\_${cross_validation_folds}.csv***: cross validation results.
- ***results\_${model_name}.csv***: final prediction results on test set.

### Improved Model Training and Evaluation
```
python run.py
```
Outputs:
- ***results\_reg\_logistic\_dynamic\_k\_${cross_validation_folds}\_poly\_${polynomial_factor}.csv***: cross validation results.
- ***results\_reg\_logistic\_dynamic\_poly\_${polynomial_factor}.csv***: final prediction results on test set (best submission).

## Results
We include the test set prediction results of all baseline models and our improved model under the folder "predictions".
