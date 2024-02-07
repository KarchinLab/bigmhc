# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------
Copyright 2023 Karchin Lab - All Rights Reserved

license: BigMHC Academic License
author: Benjamin Alexander Albert
file: mhcenc.py
------------------------------------------------------------------------

Classes to map MHC names to binary encodings.
MHCEncoder is effectively a dictionary so that all indexing is first passed
through mhcuid (e.g. __(get/set/del)item__ and __contains__).

MHCEncoder also implements fuzzy string matching to
encode MHC alleles that do not have an exact encoding.

To generate an MHCEncoder, a file of the following format is read:

  mhc,2_A,2_K,2_L,2_Q,2_R,2_W,2_X,9_F,9_H,9_I,9_L,9_P,9_R,9_T,9_V,9_X,...
  ...
  HLA-A*01:01,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0
  HLA-A*01:02,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0
  HLA-A*01:03,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0
  HLA-A*01:04,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0
  HLA-A*01:06,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0
  HLA-A*01:07,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1
  ...

In this example, HLA-A*01:07 has X (deletion) at positions 2 and 9
in the pseudosequence whereas HLA-A*01:(01-06) have A and F.

The first row names are stored in self.cols.
The subsequent rows create a mapping from MHC to encoding.
"""

import torch

from src.mhcuid import mhcuid


class MHCEncoder:

    def __init__(
        self,
        encs : dict,
        cols : list):

        self.encs = encs
        self.cols = cols
        self.subcache = dict()
        self.enclen = None

        for mhc,enc in encs.items():
            if self.enclen is None:
                self.enclen = len(enc)
            elif self.enclen != len(enc):
                raise ValueError(
                    ("MHC {} enc len {} does not"
                     " match prior enc lens {}").format(
                        mhc, len(enc), self.enclen))

    @staticmethod
    def read(mhcfile : str) -> "MHCEncoder":
        """
        Constructs an MHCEncoder from a csv file.

        Parameters
        ----------
        mhcfile : str
            path to csv file where the first column is
            the allele name and all subsequent columns
            contain values in {0,1}

        Returns
        -------
        MHCEncoder that maps each allele mhcuid to an int8 tensor
        """
        with open(mhcfile, 'r') as f:
            toks = [x.strip().split(',') for x in f.readlines()]
        encs={
            mhcuid(t[0]):torch.tensor(
                [int(v) for v in t[1:]],
                dtype=torch.int8)
            for t in toks[1:]}
        return MHCEncoder(
            encs=encs,
            cols=toks[0][1:])

    def nearestMHC(self, mhc : str) -> str:
        """
        Finds the nearest allele to the one provided.

        If an exact match is found, then it is returned.
        
        If mhc[0] is alpha and all subsequent characters
        are not alpha, then "HLA" is prepended to mhc.
        Therefore, mhcuid("A*02:01") would be become HLAA0201

        Then, we search all keys to see if one starts with
        a subset of the provided mhc chars. The subset is taken
        incrementally so that the left is preserved as the right
        is removed up to len(mhc)//2 chars are removed.
        This process is short-circuited if a key is found to start
        with all but the last char of the mhc.

        If no key has been found, then we return the closest
        allele according to the Levenshtein distance.

        All substitutes are then cached for the lifetime of the MHCEncoder.

        Parameters
        ----------
        mhc : str
            mhcuid of an allele name

        Returns
        -------
        A key in self.encs that is nearest to the mhc argument.
        """

        def _levenshtein(a, b):
            u = [x for x in range(len(b)+1)]
            v = u.copy()
            for x in range(len(a)):
                v[0] = x + 1
                for y in range(len(b)):
                    v[y+1] = min(
                        u[y+1] + 1,
                        v[y] + 1,
                        u[y] + int(a[x] != b[y]))
                tmp = u
                u = v
                v = tmp
            return u[len(b)]

        if mhc in self.encs:
            return mhc

        if mhc in self.subcache:
            return self.subcache[mhc]

        if mhc[0].isalpha() and \
                all([not mhc[x].isalpha() for x in range(1,len(mhc))]):
            mhc = "HLA" + mhc

        minpre = float("inf")
        minkey = None
        for key in self.encs:
            for x in range(1,min(minpre,len(mhc)//2)):
                if key.startswith(mhc[:-x]):
                    minpre = x
                    minkey = key
                    break
            if minpre == 1:
                break

        if minkey is None:
            minlev = float("inf")
            minkey = None
            for key in self.encs:
                lev = _levenshtein(key, mhc)
                if lev < minlev:
                    minlev = lev
                    minkey = key

        self.subcache[mhc] = minkey

        return minkey

    def keys(self):
        return self.encs.keys()

    def values(self):
        return self.encs.values()

    def items(self):
        return self.encs.items()

    def __getitem__(self, mhc):
        return self.encs.__getitem__(self.nearestMHC(mhcuid(mhc)))

    def __setitem__(self, mhc, enc):
        return self.encs.__setitem__(mhcuid(mhc), enc)

    def __delitem__(self, mhc):
        return self.encs.__delitem__(self.nearestMHC(mhcuid(mhc)))

    def __contains__(self, mhc):
        return self.encs.__contains__(mhcuid(mhc))

    def __iter__(self):
        return self.encs.__iter__()

    def __reversed__(self):
        return self.encs.__reversed__()

    def __len__(self):
        return self.encs.__len__()
