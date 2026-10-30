# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------
BigMHC Academic License

tests/testutil.py
------------------------------------------------------------------------

Shared helpers for the test suite:
  * puts src/ on sys.path so that the modules import like the scripts do
  * a verbatim copy of the original nearestMHC search (reference oracle)
  * a small synthetic pseudosequence file that exercises every rule
  * subprocess runner for predict.py / train.py

Run the suite from the repository root with:
    python -m unittest discover -s tests
"""

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
DATA = os.path.join(REPO, "data")
MODELS = os.path.join(REPO, "models")
PSEUDOSEQS = os.path.join(DATA, "pseudoseqs.csv")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

from mhcuid import mhcuid  # noqa: E402


def haveModels():
    """
    True if the EL base models (and therefore the IM heads) are present
    """
    return all([
        os.path.isdir(os.path.join(MODELS, "bat{}".format(2**x)))
        for x in range(9, 16)])


def levenshtein(a, b):
    """
    Verbatim copy of the nested _levenshtein of MHCEncoder.nearestMHC
    """
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


def referenceNearest(encs, mhc):
    """
    Reference oracle: the search of MHCEncoder.nearestMHC exactly as it
    was before conversions were recorded (commit 199e890), without the
    subcache. encs must iterate in pseudosequence file order.
    Returns the key that the original implementation returns.
    """
    if mhc in encs:
        return mhc

    if mhc[0].isalpha() and \
            all([not mhc[x].isalpha() for x in range(1,len(mhc))]):
        mhc = "HLA" + mhc

    minpre = float("inf")
    minkey = None
    for key in encs:
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
        for key in encs:
            lev = levenshtein(key, mhc)
            if lev < minlev:
                minlev = lev
                minkey = key

    return minkey


# Synthetic pseudosequence file: names in the order of the real file
# (non-human species sort before HLA), one distinct 4-bit encoding each so
# that encodings identify alleles. It is small enough to resolve thousands
# of names in a second and contains the neighbours that make the rules
# observable: HLA-A*02:01 before HLA-A*02:02, HLA-B*27:01 before
# HLA-B*27:05, HLA-C*07:01 before HLA-C*07:02, a *02:10 / *02:11 pair.
SYNTHETIC_NAMES = [
    "H2-Db",
    "H2-Kb",
    "Mamu-A1*001:01",
    "Mamu-B*001:01",
    "HLA-A*01:01",
    "HLA-A*02:01",
    "HLA-A*02:02",
    "HLA-A*02:03",
    "HLA-A*02:10",
    "HLA-A*02:11",
    "HLA-A*11:01",
    "HLA-B*07:02",
    "HLA-B*27:01",
    "HLA-B*27:05",
    "HLA-C*07:01",
    "HLA-C*07:02",
]

SYNTHETIC_COLS = ["2_A", "2_X", "9_F", "9_X"]


def syntheticEncoding(idx):
    """
    4-bit encoding of the idx-th synthetic allele, distinct per allele
    """
    return [(idx >> b) & 1 for b in range(len(SYNTHETIC_COLS))]


def writeSyntheticPseudoseqs(path):
    """
    Writes the synthetic pseudosequence file to path and returns path
    """
    with open(path, "w") as f:
        f.write("mhc," + ",".join(SYNTHETIC_COLS) + "\n")
        for idx, name in enumerate(SYNTHETIC_NAMES):
            f.write(name + "," + ",".join(
                str(v) for v in syntheticEncoding(idx)) + "\n")
    return path


def syntheticNames():
    """
    mhcuid -> name for the synthetic file
    """
    return {mhcuid(n): n for n in SYNTHETIC_NAMES}


def writeCSV(path, rows, header=None):
    """
    Writes rows (lists of str) as a comma-separated file, with optional
    header row, and returns path
    """
    with open(path, "w") as f:
        if header is not None:
            f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")
    return path


def runScript(script, args, cwd=SRC):
    """
    Runs src/<script> with args in a subprocess using the current
    interpreter and returns the CompletedProcess (stdout/stderr as str).
    stderr of DataLoader workers is captured too because they inherit
    the pipe, so duplicates printed by workers would be visible here.
    """
    cmd = [sys.executable, os.path.join(SRC, script)] + list(args)
    return subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False)


def readCSV(path):
    """
    Reads a CSV as a list of dicts with string values ("" for empty)
    """
    import csv
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def reportLines(stderr):
    """
    All lines of the conversion report in the stderr text (header and
    per-name lines). Other stderr content, e.g. deprecation warnings that
    newer pandas versions emit for the upstream code under unittest's
    warning filter, is left out.
    """
    return [l for l in stderr.splitlines() if l.startswith("MHC name conversion")]


def conversionLines(stderr):
    """
    The per-name report lines of the stderr text (header line excluded)
    """
    return [l for l in reportLines(stderr)
            if l.startswith("MHC name conversion:") or
               l.startswith("MHC name conversion WARNING:")]


def reportedInputs(stderr):
    """
    The input='...' values of the report lines, in order
    """
    out = list()
    for line in conversionLines(stderr):
        start = line.index("input='") + len("input='")
        end = line.index("' used='", start)
        out.append(line[start:end])
    return out
