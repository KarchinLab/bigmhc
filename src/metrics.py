# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------
Copyright 2023 Benjamin Alexander Albert [Karchin Lab]
All Rights Reserved

BigMHC Academic License

metrics.py
------------------------------------------------------------------------
"""

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np


def auroc(T,P):
    return roc_auc_score(T,P)


def auprc(T,P):
    ppv, tpr, _ = precision_recall_curve(T,P)
    return auc(tpr,ppv)


def meanppvn(T,P):
    tsum = int(T.sum())
    idx = np.argsort(P)[::-1]
    x = np.arange(1,tsum+1)
    return np.mean(np.cumsum(T[idx[:tsum]]) / x)
