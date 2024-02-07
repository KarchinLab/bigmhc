# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------
Copyright 2023 Benjamin Alexander Albert [Karchin Lab]
All Rights Reserved

BigMHC Academic License

dataset.py
------------------------------------------------------------------------
"""

import math
import random

import pandas as pd

import torch

from typing import Union, Callable

import src.mhcuid
import src.encseq


class Batch:
    """
    A struct to hold pMHC data and a target value.
    """
    def __init__(self, mhc, pep, tgt):
        self.mhc = mhc
        self.pep = pep
        self.tgt = tgt

    def __len__(self):
        return len(self.tgt)

    def extend(self, other):
        self.mhc += other.mhc
        self.pep += other.pep
        self.tgt += other.tgt


class Dataset:
    """
    A wrapper for a Pandas DataFrame with helper functions
    for encoding pMHCs and batching instances.
    """

    def __init__(
            self,
            pmhcs  : pd.DataFrame,
            mhcenc : dict = None,
            encpep : Callable = src.encseq.dummy):

        self.df     = pmhcs
        self.mhcenc = mhcenc
        self.encpep = encpep 
        self.bats   = None

    def makebats(
            self,
            maxbat:    int,
            shuffle:  bool = False,
            negfrac: float = 0,
            evendist: bool = False) -> None:
        """
        Populates self.bats with references to self.df rows.
        self.df index is reset to be 0,1,2,...,len(self.df)-1.

        self.bats becomes a Pandas DataFrame containing columns
        batidx and dfidx. The batidx column indicates the batch
        index to which the instance at row dfidx belongs. The
        dfidx references the row of self.df
        
        Parameters
        ----------

        maxbat : int
            Maximum batch size in number of instances

        shuffle : bool
            True to randomly shuffle self.df before and after batching

        negfrac : float
            Fraction of instances to be negative per batch.
            Set to 0 to not use.
            If negfrac is too high, then there may not be enough
            negatives to evenly distribute across the batches
            without sampling with replacement. In this case, the
            negatives are evenly distributed across the batches.

        evendist : bool
            True to make batches of equal size per peptide length grouping.
            This is only applied if not using negfrac.

        Returns
        -------
        None
        """

        self.bats = list()
        if shuffle:
            self.df = self.df.sample(frac=1)
        self.df.reset_index(inplace=True, drop=True)
        grps = self.df.groupby("len")
        if negfrac:
            for _, grp in grps:
                pos = grp.loc[grp["tgt"]==1,:]
                neg = grp.loc[grp["tgt"]==0,:]
                numbats = math.ceil(len(grp) / maxbat)
                steppos = int(math.ceil(len(pos) / numbats))
                stepneg = int(math.ceil(len(neg) / numbats))
                stepneg = int(min(stepneg, steppos / (1-negfrac) - steppos))
                idx1pos = 0
                idx2pos = steppos
                idx1neg = 0
                idx2neg = stepneg
                while idx1pos < len(pos):
                    self.bats.append(
                        list(pos.index[idx1pos:idx2pos]) +
                        list(neg.index[idx1neg:idx2neg]))
                    idx1pos = idx2pos
                    idx1neg = idx2neg
                    idx2pos = min(idx2pos + steppos, len(pos))
                    idx2neg = min(idx2neg + stepneg, len(neg))
        else:
            for _, grp in grps:
                if evendist:
                    numbats = math.ceil(len(grp) / maxbat)
                    step = int(math.ceil(len(grp) / numbats))
                else:
                    step = maxbat
                idx1 = 0
                idx2 = min(step, len(grp)) 
                while idx1 < len(grp):
                    self.bats.append(list(grp.index[idx1:idx2]))
                    idx1 = idx2
                    idx2 = min(idx2 + step, len(grp))
        batidx = [0 for _ in range(len(self.bats[0]))]
        for x in range(1,len(self.bats)):
            self.bats[0].extend(self.bats[x])
            batidx.extend([x for _ in range(len(self.bats[x]))])
        self.bats = pd.DataFrame(
            {"batidx":batidx,
             "dfidx" :self.bats[0]}).groupby("batidx")

    def __len__(self):
        return len(self.bats)

    def __getitem__(self, idx : int) -> Batch:
        return self.getbat(idx=idx, enc=True)

    def getbat(
        self,
        idx : int,
        enc : bool) -> Union[pd.DataFrame, Batch]:
        """
        Get the Batch at the requested index with optional encoding.

        Parameters
        ----------

        idx : int
            index of the batch as found in self.bats in the batidx column

        enc : bool
            True to encode:
                each MHC using self.mhcenc
                each peptide using self.encpep
                the target values as a torch Tensor

        Returns
        -------
        The batch as either a Pandas DataFrame if enc is False
        or encoded as torch Tensors stored in a Batch object
        """

        if idx >= len(self.bats):
            raise ValueError(
                "idx ({}) > num bats ({})".format(idx, len(self.bats)))
        bat = self.df.iloc[self.bats.get_group(idx)["dfidx"],:].copy()
        if not enc:
            return bat
        mhc = torch.stack(bat["mhc"].apply(self.mhcenc.__getitem__).tolist())
        pep = torch.stack(bat["pep"].apply(self.encpep).tolist())
        tgt = torch.tensor(bat["tgt"].values)
        return Batch(
            mhc=mhc,
            pep=pep,
            tgt=tgt)

    @staticmethod
    def readPMHCs(
            fp      : str,
            allele  : Union[str, int],
            pepcol  : int,
            tgtcol  : int  = None,
            hdrcnt  : int  = 0,
            delim   : str  = ',',
            verbose : bool =True) -> pd.DataFrame:
        """
        Reads a DataFrame of pMHC instances with optional measurements.

        Parameters
        ----------

        fp : str
            filepath to delimited input data

        allele : str, int
            MHC str or index of column containing MHC alleles

        pepcol : int
            column index containing peptides

        tgtcol : int, default=None
            target column (e.g. BA/EL/IM data)

        hdrcnt : int, default=0
            number of rows to skip from the start of the file

        delim : str, defualt=','
            column delimiter

        Returns
        -------
        DataFrame with columns:
            uid (index)
            mhc (string)
            pep (string)
            tgt (float32)
            len (int8)

        Where the uid index is of the format: mhc_pep
        """

        if verbose:
            print("processing {}...".format(fp), end="", flush=True)

        df = pd.read_csv(
            fp,
            sep=delim,
            skiprows=hdrcnt,
            header=None,
            engine='c')

        # insert allele as column if str, else rename allele col to "mhc"
        if isinstance(allele, str):
            df["mhc"] = allele
        else:
            df.rename(
                columns={df.columns[allele]:"mhc"},
                inplace=True)

        # make tgt column with single value if tgtcol is not provided
        if tgtcol is None or tgtcol < 0:
            df["tgt"] = None
        
        # standardize column names
        df.rename(
            columns={df.columns[pepcol]:"pep"},
            inplace=True)

        df["pep"] = df["pep"].apply(str.upper)

        if tgtcol is not None:
            df.rename(
                columns={df.columns[tgtcol]:"tgt"},
                inplace=True)

        # remove unneeded columns and reorder columns
        df = df[["mhc","pep","tgt"]]

        # create column of peptide lengths
        df["len"] = df["pep"].apply(len)

        # make pMHC unique identifier index as mhcuid(mhc)_pep
        df.insert(
            loc=0,
            column="uid",
            value=df["mhc"].apply(src.mhcuid.mhcuid) + '_' + df["pep"])
        df.set_index(keys="uid", drop=True, inplace=True)

        if verbose:
            print("constructed {} rows".format(len(df)))

        df = df.astype(
            {"mhc":"string",
             "pep":"string",
             "tgt":"float32",
             "len":"int8"})

        return df
