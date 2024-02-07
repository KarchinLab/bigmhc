# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------
Copyright 2023 Benjamin Alexander Albert [Karchin Lab]
All Rights Reserved

BigMHC Academic License

cli.py
------------------------------------------------------------------------

Defines an ArgumentParser for predict.py and train.py CLIs along with
helper functions for loading BigMHC models, accelerating them on GPUs,
and loading data. Because predict.py and train.py share many common
CLI arguments and their model and data loading are the same, the shared
functions are defined here instead of duplicating in each executable script.
"""

import argparse
import random
import os
import time
import psutil

import numpy as np
import torch

from typing import Union

from src.bigmhc import BigMHC

from src.dataset import Dataset
from src.mhcenc import MHCEncoder


def _rootdir():
    return \
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(
                    __file__)))


def _parseTransferLearn(args : argparse.Namespace) -> argparse.Namespace:
    """
    If model is provided, then set args.transferlearn to True
    otherwise set args.transferlearn to False
    """
    args.transferlearn = (args.train and (args.models is not None))
    return args


def _parseModel(args : argparse.Namespace) -> argparse.Namespace:
    """
    Replace "el" with "bigmhc_el" and "im" with "bigmhc_im".
    Then replace "bigmhc_el" and "bigmhc_im" with relative paths
    Lastly, convert relative paths to absolute paths so that
    using os.path.basename is more reliable
    """
    if args.models is None:
        if not args.train:
            raise ValueError("Prediction requires models to be set")
        return args 

    elmodels = [
            os.path.join(
                _rootdir(),
                "models",
                "bat{}".format(int(2**x)))
        for x in range(9,16)]

    immodels = [os.path.join(x, "im") for x in elmodels]

    models = list()
    if args.models.strip().lower() == "el":
        args.modelname = "BigMHC_EL"
        models = elmodels
    elif args.models.strip().lower() == "im":
        args.modelname = "BigMHC_IM"
        models = immodels
    else:
        args.modelname = "BigMHC"
        for path in args.models.split(':'):
            models.append(os.path.abspath(path))
    args.models = models
    return args


def _parseDevices(args : argparse.Namespace) -> argparse.Namespace:
    """
    convert "cpu" into an empty list
    convert "all" into a list of all device indices
    convert comma-separated string of device indices into a list of ints
    """
    args.devices = args.devices.lower()
    if args.devices == "cpu":
        args.devices = []
    else:
        if not torch.cuda.is_available():
            raise ValueError(
                "Torch CUDA is not available. To run on CPU, set devices=cpu")
        if args.devices == "all":
            args.devices = [x for x in range(torch.cuda.device_count())]
        else:
            args.devices = [int(d) for d in args.devices.split(',')]
    return args


def _parseMaxbat(args : argparse.Namespace) -> argparse.Namespace:
    """
    An RTX 3090 with 24GB of global memory can use a batch size of 18000
    for making predictions, so we linearly scale this maximum batch size
    according to the minimum global memory of all requested devices.
    If using the CPU, then the total virtual memory is used for scaling.
    """
    if args.maxbat is not None and args.maxbat > 0:
        return args
    if len(args.devices):
        minmem = min([torch.cuda.get_device_properties(d).total_memory
            for d in args.devices])
    else:
        minmem = psutil.virtual_memory().total
    args.maxbat = int((minmem / (1024**3)) / 24 * 18000)
    if len(args.devices):
        args.maxbat *= len(args.devices)
    return args


def _parseOut(args : argparse.Namespace) -> argparse.Namespace:
    """
    If transfer learning, out defaults to the models path.
    If training from scratch, then an out dir must be set.
    """
    if args.transferlearn:
        if args.out is None:
            args.out = args.models[0]
    elif args.train and args.out is None:
        raise ValueError(
            "training from scratch requires a out arg to be set")
    return args


def _parseJobs(args : argparse.Namespace) -> argparse.Namespace:
    if args.jobs is None or args.jobs <= 0:
        args.jobs = max(1, psutil.cpu_count(logical=False) // 2)
    return args


def _parsePrefetch(args : argparse.Namespace) -> argparse.Namespace:
    if args.prefetch is None or args.prefetch <= 0:
        args.prefetch = 16
    return args


def _parseHdrcnt(args : argparse.Namespace) -> argparse.Namespace:
    if args.hdrcnt >= 0:
        return args
    raise ValueError("hdrcnt must be nonnegative")


def _parseAllele(args : argparse.Namespace) -> argparse.Namespace:
    if isinstance(args.allele, str) and all([x.isdigit() for x in args.allele]):
        args.allele = int(args.allele)
    return args


def _parsePepcol(args : argparse.Namespace) -> argparse.Namespace:
    if args.pepcol >= 0:
        return args
    raise ValueError("pepcol must be nonnegative")


def _parseTgtcol(args : argparse.Namespace) -> argparse.Namespace:
    if args.tgtcol is None or args.tgtcol >= 0:
        return args
    raise ValueError("tgtcol must be nonnegative")


def _loadModels(args : argparse.Namespace) -> BigMHC:
    """
    Wrapper for BigMHC.load with optional verbose printing.
    Additionally sets model eval or train mode.
    """

    if args.models is None:
        if args.verbose:
            print("creating new model...", end="")
        models = [BigMHC()]
        if args.verbose:
            print("done")
    else:
        models = list()
        for model in args.models:
            if args.verbose:
                print("loading model {}...".format(model), end="")
            models.append(BigMHC.load(model))
            if args.verbose:
                print("done")
    return models


def _loadData(args : argparse.Namespace) -> Dataset:
    """
    Constructs and returns a Dataset object
    """
    mhcenc = MHCEncoder.read(args.pseudoseqs)
    pmhcs = Dataset.readPMHCs(
        fp=args.input,
        allele=args.allele,
        pepcol=args.pepcol,
        tgtcol=args.tgtcol,
        hdrcnt=args.hdrcnt,
        verbose=args.verbose)
    data = Dataset(
        pmhcs=pmhcs,
        mhcenc=mhcenc)
    data.makebats(
        args.maxbat,
        shuffle=args.train,
        evendist=args.transferlearn)
    data = torch.utils.data.DataLoader(
        data,
        batch_size=None,
        shuffle=args.train,
        num_workers=args.jobs,
        prefetch_factor=args.prefetch,
        persistent_workers=True)
    return data


def parseArgs(train):

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(
        ("Train BigMHC from scratch or just the final and penultimate layers"
         " using a specified base model for transfer learning") if train else
         "Predict pMHC presentation or immunogenicity with BigMHC")

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="path to csv file")

    parser.add_argument(
        "-m",
        "--models",
        type=str,
        required=(not train),
        help=("Either \"el\" or \"im\" for presentation (eluted ligand)"
              " and immunogenicity prediction respectively."
              " Or specify a colon-delimited paths to model directories"))

    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default=None,
        help="Path to dir in which to save trained model" if train else
             ("Path to file in which to save predictions."
              " If None, then predictions are saved to [INPUT].prd"))

    parser.add_argument(
        "-s",
        "--pseudoseqs",
        type=str,
        default=os.path.join(_rootdir(), "data", "pseudoseqs.csv"),
        help="CSV file mapping MHC to one-hot encoding")

    parser.add_argument(
        "-d",
        "--devices",
        type=str,
        default="all",
        help=("Comma-separated list of GPU device ids."
              " Use \"all\" to use all GPUs or \"cpu\" to use the CPU"))

    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=True,
        help="1 to enable verbose printing, 0 otherwise")

    parser.add_argument(
        "-b",
        "--maxbat",
        type=int,
        default=32 if train else None,
        help=("Maxmum batch size (turn down if running out of memory)."
              " If None, we guess the max by the devices arg."))

    parser.add_argument(
        "-c",
        "--hdrcnt",
        type=int,
        default=1,
        help="Number of header lines to skip in the input file")

    parser.add_argument(
        "-a",
        "--allele",
        default=0,
        help=("Zero-indexed column of mhc alleles or an allele name."
              " For example, use 0 if the first column of the input file"
              " contains an MHC allele. You can specify an allele name"
              " to apply a specified allele to all peptides by passing"
              " an allele name in the format: HLA-A*02:01"))

    parser.add_argument(
        "-p",
        "--pepcol",
        type=int,
        default=1,
        help="Zero-indexed column of the input file containing peptides")

    parser.add_argument(
        "-t",
        "--tgtcol",
        type=int,
        default=None,
        required=train,
        help="Zero-indexed column of the input file containing target values")

    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="Number of workers for parallel data loading")

    parser.add_argument(
        "-f",
        "--prefetch",
        type=int,
        default=None,
        help="Number of batches to prefetch per data loader worker")

    if train:

        parser.add_argument(
            "-l",
            "--lr",
            type=float,
            default=5e-5,
            help="Optimizer learning rate")

        parser.add_argument(
            "-e",
            "--epochs",
            type=int,
            default=50,
            help="Number of training epochs")

    else:

        parser.add_argument(
            "-z",
            "--saveatt",
            type=int,
            default=False,
            help="1 to save attention values when running predict, 0 otherwise")

    args = parser.parse_args()

    args.train = train

    args = _parseTransferLearn(args)
    args = _parseModel(args)
    args = _parseDevices(args)
    args = _parseMaxbat(args)
    args = _parseOut(args)

    args = _parseHdrcnt(args)
    args = _parseAllele(args)
    args = _parsePepcol(args)
    args = _parseTgtcol(args)

    args = _parseJobs(args)
    args = _parsePrefetch(args)

    data  = _loadData(args)
    models = _loadModels(args)

    return args, data, models
