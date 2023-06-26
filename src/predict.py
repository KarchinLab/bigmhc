#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------
Copyright 2023 Benjamin Alexander Albert [Karchin Lab]
All Rights Reserved

BigMHC Academic License

predict.py
------------------------------------------------------------------------
"""

import os

import torch
import pandas as pd

import cli 

from bigmhc import BigMHC


def predict(models, data, args):

    if args.verbose:
        print("starting prediction devices: {}".format(args.devices))

    for x in range(len(models)):
        models[x].eval()
        models[x] = BigMHC.accelerate(
            models[x],
            devices=args.devices)

    preds = list()
    with torch.no_grad():
        for idx,bat in enumerate(data):
            if args.verbose:
                print("batch {}/{}".format(idx+1, len(data)))
            out = list()
            att = list()
            for model in models:
                dev = next(model.parameters()).device
                _out,_att = model(
                    mhc=bat.mhc.to(dev),
                    pep=bat.pep.to(dev))
                out.append(torch.sigmoid(_out))
                att.append(_att)
            out = torch.mean(torch.stack(out),dim=0)
            att = torch.mean(torch.stack(att),dim=0)
            rawbat = data.dataset.getbat(idx=idx, enc=False)
            rawbat[args.modelname] = out.cpu().numpy()
            if args.saveatt:
                attdict = dict()
                for x in range(att.shape[1]):
                    attdict["att_{}".format(x)] = att[:,x].cpu()
                attdf = pd.DataFrame(attdict, index=rawbat.index)
                rawbat = pd.concat((rawbat, attdf), axis=1)
            preds.append(rawbat)
    return pd.concat(preds).sort_index()


def main():

    args, data, models = cli.parseArgs(train=False)

    preds = predict(models=models, data=data, args=args)

    if not args.out:
        args.out = args.input + ".prd"

    if args.verbose:
        print("writing predictions to {}".format(args.out))

    if args.tgtcol is not None:
        preds["tgt"] = preds["tgt"].astype(int)

    preds.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
