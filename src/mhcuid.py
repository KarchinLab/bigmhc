# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------
Copyright 2022 Benjamin Alexander Albert [Karchin Lab]
All Rights Reserved

BigMHC Academic License

mhcuid.py
------------------------------------------------------------------------

Generates unique identifier for a given MHC allele.

Removes all non-alphanumeric chars to unify allele formats.
Converts all lower-case letters to upper-case.
e.g. "hla-A*02:01" becomes HLAA0201
"""


import re


def mhcuid(mhc : str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", mhc.upper())
