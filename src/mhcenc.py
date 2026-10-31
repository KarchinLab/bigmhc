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
MHCEncoder.resolveMHC reports which allele of the file was actually used
for a given name and which conversions led to it (see MHCResolution).

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
The allele names as written in the file are kept in self.names
(mhcuid -> name) so that conversions can be reported in a readable form.
"""

import torch

from mhcuid import mhcuid


class MHCResolution:
    """
    A record of how one allele name, as provided by the user,
    was resolved to an entry of the pseudosequence file.

    mhc : str
        allele name exactly as provided
    key : str
        mhcuid(mhc), the normalized form that is looked up
    usedkey : str
        key in MHCEncoder.encs whose encoding is used for mhc
    used : str
        name of that entry as written in the pseudosequence file
    steps : list
        conversions applied, in order. Empty when mhc is written exactly
        as in the file. Otherwise one or more of:
            "normalization"       the normalized name is in the file but
                                  mhc is written differently (case or
                                  punctuation); never combined with others
            "HLA prefix"          "HLA" was prepended before searching
            "truncation (x=N)"    the entry starting with all but the last
                                  N chars of the (prefixed) name was used
            "edit distance (d=N)" the entry with the smallest Levenshtein
                                  distance N was used; this is a name
                                  match only and says nothing about the
                                  biological similarity of the two alleles
    """

    def __init__(self, mhc, key, usedkey, used, steps):
        self.mhc     = mhc
        self.key     = key
        self.usedkey = usedkey
        self.used    = used
        self.steps   = steps

    @property
    def converted(self) -> bool:
        """
        True if the allele used is not written exactly as mhc
        """
        return len(self.steps) > 0

    @property
    def byEditDistance(self) -> bool:
        """
        True if the allele used was chosen by edit distance
        """
        return any([s.startswith("edit distance") for s in self.steps])

    @property
    def reason(self) -> str:
        """
        Human-readable summary of steps, e.g. "HLA prefix + truncation (x=1)"
        """
        return " + ".join(self.steps) if self.steps else "exact"

    def __repr__(self):
        return "MHCResolution(mhc={!r}, used={!r}, reason={!r})".format(
            self.mhc, self.used, self.reason)


class MHCEncoder:

    def __init__(
        self,
        encs : dict,
        cols : list,
        names : dict = None):

        self.encs = encs
        self.cols = cols
        self.names = names if names is not None else {k:k for k in encs}
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
        and remembers the allele name as written in the file
        """
        with open(mhcfile, 'r') as f:
            toks = [x.strip().split(',') for x in f.readlines()]
        encs={
            mhcuid(t[0]):torch.tensor(
                [int(v) for v in t[1:]],
                dtype=torch.int8)
            for t in toks[1:]}
        names={mhcuid(t[0]):t[0] for t in toks[1:]}
        return MHCEncoder(
            encs=encs,
            cols=toks[0][1:],
            names=names)

    def _searchMHC(self, mhc : str) -> tuple:
        """
        Fuzzy search for an mhcuid that is not a key of self.encs.
        This is the search performed by nearestMHC, shared with resolveMHC
        so that the reported conversions are the ones actually applied.

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

        Parameters
        ----------
        mhc : str
            mhcuid of an allele name that is not in self.encs

        Returns
        -------
        Tuple (mhc, key, steps) where mhc is the searched name after the
        optional "HLA" prefix, key is the nearest key in self.encs, and
        steps is the list of conversions applied (see MHCResolution).
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

        steps = list()

        if mhc[0].isalpha() and \
                all([not mhc[x].isalpha() for x in range(1,len(mhc))]):
            mhc = "HLA" + mhc
            steps.append("HLA prefix")

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
            steps.append("edit distance (d={})".format(minlev))
        else:
            steps.append("truncation (x={})".format(minpre))

        return mhc, minkey, steps

    def nearestMHC(self, mhc : str) -> str:
        """
        Finds the nearest allele to the one provided.

        If an exact match is found, then it is returned.
        Otherwise the search described in _searchMHC is performed.

        All substitutes are then cached for the lifetime of the MHCEncoder.
        Note that the cache is read with the name as provided but written
        with the name after the optional "HLA" prefix, so prefixed names
        are searched again on each call; this does not affect the result.

        Parameters
        ----------
        mhc : str
            mhcuid of an allele name

        Returns
        -------
        A key in self.encs that is nearest to the mhc argument.
        """

        if mhc in self.encs:
            return mhc

        if mhc in self.subcache:
            return self.subcache[mhc]

        mhc, minkey, _ = self._searchMHC(mhc)

        self.subcache[mhc] = minkey

        return minkey

    def resolveMHC(self, mhc : str) -> MHCResolution:
        """
        Resolves an allele name as provided by the user and records how.

        As long as the encoding table (self.encs) is not modified while
        this MHCEncoder is in use, the usedkey of the result equals
        nearestMHC(mhcuid(mhc)), so the encoding used is unchanged and
        this method only adds the record of the conversions. The search
        is repeated instead of reading self.subcache, so that the recorded
        steps are the ones that apply to this name, and self.subcache is
        left untouched; nearestMHC, in contrast, keeps returning cached
        substitutes for the lifetime of the encoder. Whenever the table
        changes, build a new MHCEncoder (e.g. with MHCEncoder.read) rather
        than modifying one that has already been used.

        Parameters
        ----------
        mhc : str
            allele name as provided by the user (not an mhcuid)

        Returns
        -------
        MHCResolution with the name as written in the pseudosequence file
        """
        key = mhcuid(mhc)
        if key in self.encs:
            usedkey = key
            steps = list()
            if mhc != self.names.get(key, key):
                steps.append("normalization")
        else:
            _, usedkey, steps = self._searchMHC(key)
        return MHCResolution(
            mhc=mhc,
            key=key,
            usedkey=usedkey,
            used=self.names.get(usedkey, usedkey),
            steps=steps)

    def keys(self):
        return self.encs.keys()

    def values(self):
        return self.encs.values()

    def items(self):
        return self.encs.items()

    def __getitem__(self, mhc):
        return self.encs.__getitem__(self.nearestMHC(mhcuid(mhc)))

    def __setitem__(self, mhc, enc):
        self.names[mhcuid(mhc)] = mhc
        return self.encs.__setitem__(mhcuid(mhc), enc)

    def __delitem__(self, mhc):
        key = self.nearestMHC(mhcuid(mhc))
        self.names.pop(key, None)
        return self.encs.__delitem__(key)

    def __contains__(self, mhc):
        return self.encs.__contains__(mhcuid(mhc))

    def __iter__(self):
        return self.encs.__iter__()

    def __reversed__(self):
        return self.encs.__reversed__()

    def __len__(self):
        return self.encs.__len__()
