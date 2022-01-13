import numpy as np
from sklearn.metrics import f1_score


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    acc = np.mean(preds == labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }
