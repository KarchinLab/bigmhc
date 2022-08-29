#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------
Copyright 2022 Benjamin Alexander Albert [Karchin Lab]
All Rights Reserved

BigMHC Academic License

predict.py
------------------------------------------------------------------------
"""

import os

import torch
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

import cli 


def printmetrics(T,P):

    def _ppvn(n):
        return T[idx[:n]].sum() / float(min(n, tsum))

    tsum = int(T.sum())
    if not tsum or tsum == len(T):
        raise ValueError("only one class found in target values")
    idx = np.argsort(P)[::-1]
    T = T[idx]
    P = P[idx]

    auppv = auc(
        np.linspace(0,1,tsum),
        np.cumsum(T[idx[:tsum]]) / np.arange(1,tsum+1))

    ppv, tpr, _ = precision_recall_curve(T,P)
    print("AUROC: {}".format(roc_auc_score(T,P)))
    print("AUPRC: {}".format(auc(tpr,ppv)))
    if tsum > 50:
        print("PPVn (top 50): {}".format(_ppvn(50)))
    print("PPVn (top {}): {}".format(tsum, _ppvn(tsum)))
    print("AUPPVn: {}".format(auppv))


def predict(model, data, args):
    if args.verbose:
        print("starting prediction devices: {}".format(args.devices))
    model.eval()
    preds = list()
    dev = next(model.parameters()).device
    modelname = os.path.basename(args.model)
    with torch.no_grad():
        for idx,bat in enumerate(data):
            if args.verbose:
                print("batch {}/{}".format(idx+1, len(data)))
            out,att = model(
                mhc=bat.mhc.to(dev),
                pep=bat.pep.to(dev))
            rawbat = data.dataset.getbat(idx=idx, enc=False)
            rawbat[modelname] = out.cpu()
            if args.saveatt:
                attdict = dict()
                for x in range(att.shape[1]):
                    attdict["att_{}".format(x)] = att[:,x].cpu()
                attdf = pd.DataFrame(attdict, index=rawbat.index)
                rawbat = pd.concat((rawbat, attdf), axis=1)
            preds.append(rawbat)
    preds = pd.concat(preds).sort_index()
    if not args.out:
        args.out = args.input + ".prd"
    if args.verbose:
        print("writing predictions to {}".format(args.out))
    preds.to_csv(args.out, index=False)

    if args.tgtcol is not None:
        printmetrics(preds["tgt"], preds[modelname])


def main():

    args, data, model = cli.parseArgs(retrain=False)

    predict(model=model, data=data, args=args)


if __name__ == "__main__":
    main()

