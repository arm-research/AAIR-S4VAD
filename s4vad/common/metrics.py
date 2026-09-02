# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
from typing import List
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


def auroc_score(targets: List, probs: List):
    y_true = np.array(targets)
    y_pred = np.array(probs)
    auroc = roc_auc_score(y_true, y_pred)
    return auroc

def f1_acc_score(targets: List, probs: List, threshold: float = 0.5):
    y_true = np.array(targets)
    y_prob = np.array(probs)
    # Threshold
    y_pred = (y_prob >= threshold).astype(int)
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Extract FN and FP using .ravel()
    TN, FP, FN, TP = cm.ravel()
    f1 = 2.0 * TP / (2 * TP + FP + FN)
    acc = (TP + TN) / (TP + TN + FP + FN)
    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    return f1, acc, fpr, tpr

def get_fpr_fnr_for_th(fpr, fnr, thresholds, threshold):
    idx = np.argmin(np.abs(thresholds - threshold))
    return fpr[idx], fnr[idx]

def get_fpr_for_fnr(_fpr, _fnr, target_fnr):
    """Get FPR value for a specific FNR"""
    # Find the closest FNR to target
    idx = np.argmin(np.abs(_fnr - target_fnr))
    return _fpr[idx], _fnr[idx]

def get_fnr_for_fpr(_fpr, _fnr, target_fpr):
    """Get FNR value for a specific FPR"""
    # Find the closest FPR to target
    idx = np.argmin(np.abs(_fpr - target_fpr))
    return _fnr[idx], _fpr[idx]

def get_fpr_for_th(fpr, fnr, thresholds, threshold):
    idx = np.argmin(np.abs(thresholds - threshold))
    return fpr[idx], fnr[idx]