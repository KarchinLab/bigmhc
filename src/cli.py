# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------
Copyright 2022 Benjamin Alexander Albert [Karchin Lab]
All Rights Reserved

BigMHC Academic License

cli.py
------------------------------------------------------------------------

Defines an ArgumentParser for predict.py and retrain.py CLIs along with
helper functions for loading BigMHC models, accelerating them on GPUs,
and loading data. Because predict.py and retrain.py share many common
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

from bigmhc import BigMHC
from dataset import Dataset
from mhcenc import MHCEncoder


def _parseModel(args : argparse.Namespace) -> argparse.Namespace:
    """
    Replace "el" with "bigmhc_el" and "im" with "bigmhc_im".
    Then replace "bigmhc_el" and "bigmhc_im" with relative paths
    Lastly, convert relative paths to absolute paths so that
    using os.path.basename is more reliable
    """
    if args.model == "el" or args.model == "im":
        args.model = "bigmhc_" + args.model
    if args.model == "bigmhc_el" or args.model == "bigmhc_im":
        args.model = os.path.join("..", "models", args.model)
    args.model = os.path.abspath(args.model)
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
    An RTX 3090 with 24GB of global memory can use a batch size of 16384
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
    args.maxbat = int((minmem / (1024**3)) / 24 * 16384)
    return args

def _parseOut(args : argparse.Namespace) -> argparse.Namespace:
    """
    if retraining without specifying an output directory
    default to a new directory: retrain_<basemodel>_<time>
    """
    if args.retrain and not args.out:
        args.out = os.path.join(
            "..",
            "models",
            "retrain_{}_{}".format(
                os.path.basename(args.model),
                int(time.time())))
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


def _accelerateModel(
    model : Union[BigMHC, torch.nn.parallel.DataParallel],
    args : argparse.Namespace) -> Union[BigMHC, torch.nn.parallel.DataParallel]:
    """
    Based on args.devices, model is sent to either the CPU, a single GPU,
    or multiple GPUs using Torch DataParallel. If using DataParallel,
    the model is first pushed to the first GPU in the devices list.
    """

    if args.verbose:
        print("sending model to device(s): {}".format(args.devices))
    if isinstance(model, torch.nn.parallel.DataParallel):
        model = model.module
    if not len(args.devices):
        return model.cpu()
    model.to(args.devices[0])
    if len(args.devices) == 1:
        return model
    return torch.nn.parallel.DataParallel(model, device_ids=args.devices)


def _loadModel(args : argparse.Namespace) -> BigMHC:
    """
    Wrapper for BigMHC.load with optional verbose printing.
    Additionally sets model eval or train mode.
    """
    if args.verbose:
        print("loading model {}...".format(args.model), end="")
    model = BigMHC.load(args.model)
    if args.retrain:
        model.train()
    else:
        model.eval()
    if args.verbose:
        print("done")
    return model


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
        shuffle=args.retrain,
        evendist=args.retrain)
    data = torch.utils.data.DataLoader(
        data,
        batch_size=None,
        num_workers=args.jobs,
        prefetch_factor=args.prefetch,
        persistent_workers=True)
    return data


def parseArgs(retrain):

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(
        ("Retrain the final and penultimate layers of BigMHC"
         "using a specified base model for transfer learning") if retrain else
        "Predict pMHC presentation or immunogenicity with BigMHC")

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="path to csv file")

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help=("either \"el\" or \"im\" for presentation (eluted ligand)"
              "and immunogenicity prediction respectively."
              "Or specify a path to a model directory"))

    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default=None,
        help="path to dir in which to save retrained model" if retrain else
             ("path to file in which to save predictions."
              "If None, then predictions are saved to [INPUT].prd"))

    parser.add_argument(
        "-s",
        "--pseudoseqs",
        type=str,
        default="../data/pseudoseqs.csv",
        help="csv file mapping MHC sequence to one-hot encoding")

    parser.add_argument(
        "-d",
        "--devices",
        type=str,
        default="all",
        help=("comma-separated list of GPU device ids."
              "Use \"all\" to use all GPUs or \"cpu\" to use the CPU"))

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
        default=1024 if retrain else None,
        help="maxmum batch size (turn down if running out of memory)")

    parser.add_argument(
        "-c",
        "--hdrcnt",
        type=int,
        default=1,
        help="number of header lines to skip in the input file")

    parser.add_argument(
        "-a",
        "--allele",
        default=0,
        help=("zero-indexed column of mhc alleles or an allele name."
              "For example, use 0 if the first column of the input file"
              "contains an MHC allele. You can specify an allele name"
              "to apply a specified allele to all peptides by passing"
              "an allele name in the format: HLA-A*02:01"))

    parser.add_argument(
        "-p",
        "--pepcol",
        type=int,
        default=1,
        help="zero-indexed column of the input file containing peptides")

    parser.add_argument(
        "-t",
        "--tgtcol",
        type=int,
        default=None,
        required=retrain,
        help=("zero-indexed column of the input file containing target values."
              "If making predictions, then model performance metrics"
              "are calculated using these target values as ground truth."))

    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="number of workers for parallel data loading")

    parser.add_argument(
        "-f",
        "--prefetch",
        type=int,
        default=None,
        help="number of batches to prefetch per data loader worker")

    if retrain:

        parser.add_argument(
            "-l",
            "--lr",
            type=float,
            default=1e-4,
            help="optimizer learning rate")

        parser.add_argument(
            "-e",
            "--epochs",
            type=int,
            default=50,
            help="number of retraining epochs")

    else:

        parser.add_argument(
            "-z",
            "--saveatt",
            type=int,
            default=False,
            help="1 to save attention values when running predict, 0 otherwise")

    args = parser.parse_args()

    args.retrain = retrain

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
    model = _loadModel(args)
    model = _accelerateModel(model, args)

    return args, data, model
