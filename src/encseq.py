# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------
Copyright 2022 Benjamin Alexander Albert [Karchin Lab]
All Rights Reserved

BigMHC Academic License

encseq.py
------------------------------------------------------------------------

Encode protein sequences as numerical vectors
"""

import torch


def dummy(seq : str) -> torch.Tensor:
    """
    Encodes a sequence by dummy-encoding amino acid chars
    and repesenting all other chars (e.g. 'X') as zeros.

    Parameters
    ----------
    seq : str
        sequence string containing only amino acid chars
        
    Returns
    -------
    Tensor of length 20*len(seq) with elements in {0,1}
    """
    res = torch.zeros(20*len(seq), dtype=torch.int8)
    idx = 0
    for r in seq.upper():
        if   r=='A': res[idx+ 0] = 1
        elif r=='C': res[idx+ 1] = 1
        elif r=='D': res[idx+ 2] = 1
        elif r=='E': res[idx+ 3] = 1
        elif r=='F': res[idx+ 4] = 1
        elif r=='G': res[idx+ 5] = 1
        elif r=='H': res[idx+ 6] = 1
        elif r=='I': res[idx+ 7] = 1
        elif r=='K': res[idx+ 8] = 1
        elif r=='L': res[idx+ 9] = 1
        elif r=='M': res[idx+10] = 1
        elif r=='N': res[idx+11] = 1
        elif r=='P': res[idx+12] = 1
        elif r=='Q': res[idx+13] = 1
        elif r=='R': res[idx+14] = 1
        elif r=='S': res[idx+15] = 1
        elif r=='T': res[idx+16] = 1
        elif r=='V': res[idx+17] = 1
        elif r=='W': res[idx+18] = 1
        elif r=='Y': res[idx+19] = 1
        idx += 20
    return res
