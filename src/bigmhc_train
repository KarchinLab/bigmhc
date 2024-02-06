#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------
Copyright 2023 Benjamin Alexander Albert [Karchin Lab]
All Rights Reserved

BigMHC Academic License

train.py
------------------------------------------------------------------------
"""

import os

import torch

import cli

from bigmhc import BigMHC


def train(model, data, args):

    if args.verbose:
        print("starting training on devices: {}".format(args.devices))

    model = BigMHC.accelerate(
        model=model,
        devices=args.devices).train()

    dev = next(model.parameters()).device
    lossf = torch.nn.BCEWithLogitsLoss().to(dev)
    
    if args.transferlearn:
        if isinstance(model, torch.nn.DataParallel):
            module = model.module
        else:
            module = model
        optmz = torch.optim.AdamW(
            params=[v for k,v in module.named_parameters()
                if k in BigMHC.tllayers()],
            lr=args.lr)
    else:
        optmz = torch.optim.AdamW(
            params=model.parameters(),
            lr=args.lr)

    for ep in range(args.epochs):
        eperr = 0
        data.dataset.makebats(
            maxbat=args.maxbat,
            shuffle=True,
            negfrac=None if args.transferlearn else 0.99)

        for bat in data:

            tgt = bat.tgt.float().to(dev)
            if args.transferlearn and not tgt.sum():
                continue

            optmz.zero_grad()
            out,_ = model(
                mhc=bat.mhc.float().to(dev),
                pep=bat.pep.float().to(dev))
            err = lossf(out, tgt)
            err.backward()
            eperr += float(err) / len(data)
            optmz.step()

        if args.verbose:
            print("ep {} loss: {}".format(ep+1, eperr))

        if args.out:
            epdir = os.path.join(args.out, "ep{}".format(ep+1))
            model = BigMHC.decelerate(model)
            model.save(epdir, tl=args.transferlearn)
            model = BigMHC.accelerate(model, args.devices)


def main():

    args, data, model = cli.parseArgs(train=True)

    if len(model) > 1:
        raise ValueError(
            "training multiple models is currently unsupported")

    train(model=model[0], data=data, args=args)


if __name__ == "__main__":
    main()
