import numpy as np
import pandas as pd
from scipy.sparse.csr import csr_matrix
from sklearn.linear_model import Ridge, LogisticRegression
from typing import Tuple, Union
from .logging_utils import common_logging


def _get_init_data(train, val, kfold, train_labels, extra_params):
    preds_train = np.zeros(train.shape[0])
    model_lst = []
    if val is not None:
        if extra_params['objective']=='multy':
            n_classes = extra_params['n_classes']
            preds_val = np.zeros([kfold.n_splits, val.shape[0], n_classes])
            preds_train = np.zeros([train.shape[0], n_classes])
            return (preds_train, model_lst, preds_val)
        preds_val = np.zeros([kfold.n_splits, val.shape[0]])
        return (preds_train, model_lst, preds_val)
    return (preds_train, model_lst, None)



@common_logging
def run_lr_log(train: csr_matrix, val: csr_matrix, train_labels: Union[pd.DataFrame, pd.Series],
               val_labels: Union[pd.DataFrame, pd.Series], lr_params: dict,
               log_params: dict, kfold, metric, extra_params) -> Tuple[np.ndarray]:
    """
    Runs LR in KFold cycle

    Simple runner. One train, one test.
    :param train: dataset to pass in kfold
    :param val: dataset to predict (no usage in training)
    :return: preds_train, preds_val
    """
    preds_train, model_lst, preds_val = _get_init_data(train, val, kfold, train_labels, extra_params)
    lr_model = LogisticRegression(**lr_params)
    for i_fold, (train_index, test_index) in enumerate(kfold.split(train)):
        lr_model.fit(train[train_index], train_labels.iloc[train_index].values)
        model_lst.append(lr_model)
        preds_train[test_index[:, np.newaxis], lr_model.classes_] = lr_model.predict_proba(train[test_index])
        if val is not None:
            row_slice = np.array(range(preds_val.shape[1])).reshape(-1, 1)
            preds_val[i_fold, row_slice, lr_model.classes_] = lr_model.predict_proba(val)
    return (preds_train, preds_val, model_lst)


@common_logging
def run_lr_reg(train: csr_matrix, val: csr_matrix, train_labels: Union[pd.DataFrame, pd.Series],
            val_labels: Union[pd.DataFrame, pd.Series], lr_params: dict,
            log_params: dict, kfold, metric, extra_params) -> Tuple[np.ndarray]:
    """
    Runs LR in KFold cycle

    Simple runner. One train, one test.
    :param train: dataset to pass in kfold
    :param val: dataset to predict (no usage in training)
    :return: preds_train, preds_val
    """
    preds_train, model_lst, preds_val = _get_init_data(train, val, kfold, train_labels, extra_params)
    lr_model = Ridge(**lr_params)
    for i_fold, (train_index, test_index) in enumerate(kfold.split(train)):
        lr_model.fit(train[train_index], train_labels.iloc[train_index].values)
        model_lst.append(lr_model)
        preds_train[test_index] = lr_model.predict(train[test_index])
        if val is not None:
            preds_val[i_fold, :] = lr_model.predict(val)
    return (preds_train, preds_val, model_lst)