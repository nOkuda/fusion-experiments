import importlib
from pathlib import Path

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import learning


def run(exp_dir, y_transformer, categories):
    data_dir = Path(__file__).parent / 'data'
    for features in ['bag_of_words', 'scores_only', 'bag_and_scores']:
        print('#', exp_dir)
        print('#', features)
        features_module = importlib.import_module(features)
        if features == 'bag_of_words':
            features_data_dir = data_dir / features
        else:
            features_data_dir = data_dir / 'phrase'
        X, y = features_module.get_dataset(features_data_dir)
        # convert from 5-scale to whatever is wanted
        y = y_transformer(y)
        models = get_models(sparse=features.startswith('bag'))
        evaluate(X, y, categories, models, features, exp_dir)
        if features == 'scores_only':
            run_per_scores(X, y, categories, exp_dir)


def run_per_scores(X, y, categories, exp_dir):
    per_scores = {
        'lemmata_only': [0],
        'semantic_only': [1],
        'sem_lem_only': [2],
        'lem+sem': [0, 1],
        'lem+sem_lem': [0, 2],
        'sem+sem_lem': [1, 2]
    }
    for label, cols in per_scores.items():
        print('#', exp_dir)
        print('#', label)
        models = get_models()
        evaluate(X[:, cols], y, categories, models, label, exp_dir)


def get_models(sparse=False):
    not_sparse = not sparse
    return [
        ('logistic_regression',
         make_pipeline(
             SimpleImputer(missing_values=np.nan, strategy='constant'),
             LogisticRegression(penalty='l2',
                                solver='lbfgs',
                                multi_class='multinomial',
                                max_iter=1000))),
        ('weighted_logistic_regression',
         make_pipeline(
             SimpleImputer(missing_values=np.nan, strategy='constant'),
             LogisticRegression(penalty='l2',
                                solver='lbfgs',
                                multi_class='multinomial',
                                max_iter=1000,
                                class_weight='balanced'))),
        ('neural_network',
         make_pipeline(
             SimpleImputer(missing_values=np.nan, strategy='constant'),
             MLPClassifier(hidden_layer_sizes=(50, ), max_iter=200000))),
        ('normalized_neural_network',
         make_pipeline(
             SimpleImputer(missing_values=np.nan, strategy='constant'),
             StandardScaler(with_mean=not_sparse),
             MLPClassifier(hidden_layer_sizes=(50, ), max_iter=200000))),
        ('oversampled_neural_network',
         make_pipeline(
             SimpleImputer(missing_values=np.nan,
                           strategy='constant'), SMOTE(),
             MLPClassifier(hidden_layer_sizes=(50, ), max_iter=200000))),
        ('imputed_logistic_regression',
         make_pipeline(
             SimpleImputer(missing_values=np.nan, strategy='mean'),
             LogisticRegression(penalty='l2',
                                solver='lbfgs',
                                multi_class='multinomial',
                                max_iter=1000))),
        ('imputed_weighted_logistic_regression',
         make_pipeline(
             SimpleImputer(missing_values=np.nan, strategy='mean'),
             LogisticRegression(penalty='l2',
                                solver='lbfgs',
                                multi_class='multinomial',
                                max_iter=1000,
                                class_weight='balanced'))),
        ('imputed_neural_network',
         make_pipeline(
             SimpleImputer(missing_values=np.nan, strategy='mean'),
             MLPClassifier(hidden_layer_sizes=(50, ), max_iter=200000))),
        ('imputed_oversampled_neural_network',
         make_pipeline(
             SimpleImputer(missing_values=np.nan, strategy='mean'), SMOTE(),
             MLPClassifier(hidden_layer_sizes=(50, ), max_iter=200000))),
        ('imputed_normalized_logistic_regression',
         make_pipeline(
             SimpleImputer(missing_values=np.nan, strategy='mean'),
             StandardScaler(with_mean=not_sparse),
             LogisticRegression(penalty='l2',
                                solver='lbfgs',
                                multi_class='multinomial',
                                max_iter=10000))),
        ('imputed_normalized_weighted_logistic_regression',
         make_pipeline(
             SimpleImputer(missing_values=np.nan, strategy='mean'),
             StandardScaler(with_mean=not_sparse),
             LogisticRegression(penalty='l2',
                                solver='lbfgs',
                                multi_class='multinomial',
                                max_iter=10000,
                                class_weight='balanced'))),
        ('imputed_normalized_neural_network',
         make_pipeline(
             SimpleImputer(missing_values=np.nan, strategy='mean'),
             StandardScaler(with_mean=not_sparse),
             MLPClassifier(hidden_layer_sizes=(50, ), max_iter=200000))),
        ('imputed_normalized_oversampled_neural_network',
         make_pipeline(
             SimpleImputer(missing_values=np.nan, strategy='mean'),
             StandardScaler(with_mean=not_sparse), SMOTE(),
             MLPClassifier(hidden_layer_sizes=(50, ), max_iter=200000))),
    ]


def evaluate(X, y, categories, models, identifier, data_dir):
    data_dir = data_dir / identifier
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
    resultspath = data_dir / f'results.{identifier}.txt'
    PRECISION = 3
    with open(str(resultspath), 'w') as ofh:
        for modelname, model in models:
            print(modelname)
            conf_mat = learning.get_confusion_matrix(data_dir, identifier,
                                                     modelname, model, X, y)
            learning.save_confusion_matrix(conf_mat, categories, data_dir,
                                           identifier, modelname)
            accuracy = np.format_float_positional(
                learning.compute_accuracy(conf_mat), precision=PRECISION)
            mcc = np.format_float_positional(
                learning.compute_matthews_corrcoef(conf_mat),
                precision=PRECISION)
            f1_score = np.format_float_positional(
                learning.compute_f1(conf_mat), precision=PRECISION)
            ofh.write(f'{modelname}\t{accuracy}\t{f1_score}\t{mcc}\n')
