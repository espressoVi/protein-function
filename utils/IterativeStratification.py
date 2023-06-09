import numpy as np
from time import perf_counter

def IterativeStratification(labels, n_splits):
    _time_start = perf_counter()
    print("Stratifying...",end = '\r')
    r = np.asarray([1 / n_splits] * n_splits)
    n_samples = labels.shape[0]
    test_folds = np.zeros(n_samples, dtype=int)
    labels_not_processed_mask = np.ones(n_samples, dtype=bool)
    c_folds = r * n_samples
    c_folds_labels = np.outer(r, labels.sum(axis=0))
    while np.any(labels_not_processed_mask):
        num_labels = labels[labels_not_processed_mask].sum(axis=0)
        label_idx = np.where(num_labels == num_labels[np.nonzero(num_labels)].min())[0]
        if label_idx.shape[0] > 1:
            label_idx = label_idx[np.random.choice(label_idx.shape[0])]
        sample_idxs = np.where(np.logical_and(labels[:, label_idx].flatten(), labels_not_processed_mask))[0]
        for sample_idx in sample_idxs:
            label_folds = c_folds_labels[:, label_idx]
            fold_idx = np.where(label_folds == label_folds.max())[0]
            if fold_idx.shape[0] > 1:
                temp_fold_idx = np.where(c_folds[fold_idx] == c_folds[fold_idx].max())[0]
                fold_idx = fold_idx[temp_fold_idx]
                if temp_fold_idx.shape[0] > 1:
                    fold_idx = fold_idx[np.random.choice(temp_fold_idx.shape[0])]
            test_folds[sample_idx] = fold_idx
            labels_not_processed_mask[sample_idx] = False
            c_folds_labels[fold_idx, labels[sample_idx]] -= 1
            c_folds[fold_idx] -= 1
    print(f"Stratification took {perf_counter() - _time_start:.2f}s")
    return test_folds
