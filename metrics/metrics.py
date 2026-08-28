import numpy as np

from metrics.f1_score_f1_pa import get_adjust_F1PA
from metrics.vus.metrics import get_vus_scores


def calculate_selected_scores(pred_labels, gt_labels):
    adjusted_pred_labels = np.asarray(pred_labels).astype(int).copy()
    gt_labels = np.asarray(gt_labels).astype(int)

    pa_accuracy, pa_precision, pa_recall, pa_f_score = get_adjust_F1PA(
        adjusted_pred_labels,
        gt_labels,
    )
    vus_scores = get_vus_scores(adjusted_pred_labels, gt_labels, 100)

    return {
        "pa_accuracy": pa_accuracy,
        "pa_precision": pa_precision,
        "pa_recall": pa_recall,
        "pa_f_score": pa_f_score,
        "VUS_ROC": vus_scores["VUS_ROC"],
        "VUS_PR": vus_scores["VUS_PR"],
    }
