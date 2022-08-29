#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------
Copyright 2022 Benjamin Alexander Albert [Karchin Lab]
All Rights Reserved

BigMHC Academic License

retrain.py
------------------------------------------------------------------------
"""

import torch

import cli


def retrain(model, data, args):
    if args.verbose:
        print("starting retraining on devices: {}".format(args.devices))
    dev = next(model.parameters()).device
    lossf = torch.nn.BCELoss(reduction="sum").to(dev)

    if isinstance(model, torch.nn.parallel.DataParallel):
        module = model.module
    else:
        module = model
    optmz = torch.optim.Adam(
        params=[*module.condenser.att.parameters(),
                *module.condenser.out.parameters()],
        lr=args.lr,
        weight_decay=args.lr/10)

    for ep in range(args.epochs):
        eperr = 0
        ninst = 0
        data.dataset.makebats(
            maxbat=args.maxbat,
            shuffle=True)
        for idx,bat in enumerate(data):
            optmz.zero_grad()
            out,_ = model(
                mhc=bat.mhc.float().to(dev),
                pep=bat.pep.float().to(dev))
            tgt = bat.tgt.float().to(dev)
            err = lossf(out, tgt)
            err.backward()
            optmz.step()
            eperr += float(err)
            ninst += len(tgt)
        eperr /= ninst
        if args.verbose:
            print("ep {} loss: {}".format(ep+1, eperr))
    if args.out:
        if args.verbose:
            print("saving retrained model to {}".format(args.out))
        if isinstance(model, torch.nn.parallel.DataParallel):
            module = model.module
        else:
            module = model
        module.cpu().save(args.out)



def main():

    args, data, model = cli.parseArgs(retrain=True)

    retrain(model=model, data=data, args=args)


if __name__ == "__main__":
    main()
