# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------
BigMHC Academic License

tests/test_predict_e2e.py
------------------------------------------------------------------------

End-to-end runs of predict.py and train.py on the CPU with the models of
the repository (skipped if they are absent).

Checks: the README examples match their .cmp files (text columns exactly,
BigMHC_EL within an absolute tolerance of 1e-5), the output keeps its
original columns and gains mhc_used last (also after the att_* columns of
-z), the stderr report is printed once (with -v=0 and with several
DataLoader workers) and names the alleles established on the unmodified
code, both input modes work, and rows that resolve to the same allele
receive the same score.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

import testutil
from testutil import (assertSameRows, checkNoRepeatedOptions, conversionLines,
                      reportLines, reportedInputs, readCSV, runScript)

# Default options of every run. They go FIRST in the command line and a
# caller sets the worker count through the jobs argument of predict(), so
# that no option is ever given twice (runScript refuses repeated options;
# argparse would silently keep the last one).
DEVICE = "-d=cpu"

# Fixture rows (input name, peptide, expected mhc_used): 13 spellings of a
# few alleles (exact, case and punctuation variants, missing HLA prefix,
# extra field, names absent from the file, an MHC-II name, nonsense) with
# 8/9/10-mer peptides, so that batching by peptide length is exercised.
# HLA-A*02:01 and A0201 with GILGFVFTL each appear twice and XYZ123 appears
# with two peptides, giving rows=2 in the report. The expected names are
# those selected by the unmodified lookup (see REAL_TABLE_EXPECTED in
# test_mhcenc.py).
TESTSET = [
    ("HLA-A*02:01",    "GILGFVFTL",  "HLA-A*02:01"),
    ("hla-a*02:01",    "GILGFVFTL",  "HLA-A*02:01"),
    ("A0201",          "GILGFVFTL",  "HLA-A*02:01"),
    ("HLA-A*02:01:01", "GILGFVFTL",  "HLA-A*02:01"),
    ("HLA-A*02:1040",  "GILGFVFTL",  "HLA-A*02:104"),
    ("HLA-DRB1*01:01", "GILGFVFTL",  "HLA-B*13:101"),
    ("XYZ123",         "GILGFVFTL",  "HLA-A*01:23"),
    ("HLA-A*02:01",    "GILGFVFTL",  "HLA-A*02:01"),
    ("A0201",          "GILGFVFTL",  "HLA-A*02:01"),
    ("HLA-A*02:01",    "FLPSDFFPSV", "HLA-A*02:01"),
    ("A*02:01",        "FLPSDFFPSV", "HLA-A*02:01"),
    ("HLA-A02:01",     "FAYLRDLL",   "HLA-A*02:01"),
    ("HLAA0201",       "FAYLRDLL",   "HLA-A*02:01"),
    ("HLA-A*02:15N",   "FLPSDFFPSV", "HLA-A*02:150"),
    ("HLA-B*07:02",    "GILGFVFTL",  "HLA-B*07:02"),
    ("hla-drb1*01:01", "FAYLRDLL",   "HLA-B*13:101"),
    ("XYZ123",         "FLPSDFFPSV", "HLA-A*01:23"),
]

CONVERTED = ["hla-a*02:01", "A0201", "HLA-A*02:01:01", "HLA-A*02:1040",
             "HLA-DRB1*01:01", "XYZ123", "A*02:01", "HLA-A02:01", "HLAA0201",
             "HLA-A*02:15N", "hla-drb1*01:01"]

BY_EDIT_DISTANCE = ["HLA-DRB1*01:01", "XYZ123", "hla-drb1*01:01"]

EXAMPLE1 = os.path.join(testutil.DATA, "example1.csv")
EXAMPLE2 = os.path.join(testutil.DATA, "example2.csv")

CMP_COLUMNS = ["mhc", "pep", "tgt", "len"]
SCORE_TOL = 1e-5   # absolute tolerance for BigMHC_EL against the .cmp files

# Runs predict.main() exactly as predict.py does, with the DataLoader
# class replaced by a subclass that records the num_workers it was built
# with, so that a test can prove how many workers a run really used.
RECORD_WORKERS_DRIVER = textwrap.dedent("""
    import sys
    sys.path.insert(0, {src!r})
    import torch
    seen = list()
    _DataLoader = torch.utils.data.DataLoader
    class _Recording(_DataLoader):
        def __init__(self, *args, **kwargs):
            seen.append(kwargs.get("num_workers"))
            super().__init__(*args, **kwargs)
    torch.utils.data.DataLoader = _Recording
    import predict
    sys.argv = ["predict.py"] + {args!r}
    predict.main()
    print("DATALOADER_NUM_WORKERS=" + ",".join(str(n) for n in seen))
    """)


def predictArgs(args, out, jobs):
    """
    Full predict.py argument list: defaults first, caller args, output
    """
    args = [DEVICE, "-j={}".format(jobs)] + list(args) + ["-o=" + out]
    checkNoRepeatedOptions(args)
    return args


@unittest.skipUnless(testutil.haveModels(), "models/ not found")
class PredictEndToEndTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bigmhc_e2e_")
        cls.testset = testutil.writeCSV(
            os.path.join(cls.tmp, "testset.csv"),
            [(m, p) for m, p, _ in TESTSET], header=["mhc", "pep"])
        cls.testset_tgt = testutil.writeCSV(
            os.path.join(cls.tmp, "testset_tgt.csv"),
            [(m, p, i % 2) for i, (m, p, _) in enumerate(TESTSET)],
            header=["mhc", "pep", "tgt"])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def predict(self, name, args, jobs=1):
        """
        Runs predict.py with the default device, the given number of
        DataLoader workers and args; returns (process, rows, output path)
        """
        out = os.path.join(self.tmp, name + ".prd")
        proc = runScript("predict.py", predictArgs(args, out, jobs))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(os.path.exists(out))
        return proc, readCSV(out), out

    def predictRecordingWorkers(self, name, args, jobs):
        """
        Like predict(), but through RECORD_WORKERS_DRIVER; additionally
        returns the num_workers values the DataLoader was built with
        """
        out = os.path.join(self.tmp, name + ".prd")
        code = RECORD_WORKERS_DRIVER.format(src=testutil.SRC, args=predictArgs(args, out, jobs))
        proc = subprocess.run(
            [sys.executable, "-c", code], cwd=testutil.SRC,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(os.path.exists(out))
        marker = [l for l in proc.stdout.splitlines() if l.startswith("DATALOADER_NUM_WORKERS=")]
        self.assertEqual(len(marker), 1, proc.stdout)
        workers = [int(w) for w in marker[0].split("=", 1)[1].split(",")]
        return proc, readCSV(out), out, workers

    def test_example1_matches_cmp_within_tolerance_and_is_silent(self):
        proc, rows, _ = self.predict("example1", ["-i=" + EXAMPLE1, "-m=el", "-t=2"])
        cmp = readCSV(EXAMPLE1 + ".cmp")
        self.assertEqual(list(rows[0].keys()), list(cmp[0].keys()) + ["mhc_used"])
        assertSameRows(rows, cmp, CMP_COLUMNS, scorecol="BigMHC_EL", tol=SCORE_TOL,
                       label="example1.prd", reflabel="example1.csv.cmp")
        self.assertEqual([r["mhc_used"] for r in rows], [r["mhc"] for r in rows])
        self.assertEqual(reportLines(proc.stderr), [])

    def test_example2_fixed_allele_matches_cmp_within_tolerance_and_is_silent(self):
        proc, rows, _ = self.predict(
            "example2", ["-i=" + EXAMPLE2, "-m=el", "-a=HLA-A*02:02", "-p=0", "-c=0"])
        cmp = readCSV(EXAMPLE2 + ".cmp")
        self.assertEqual(list(rows[0].keys()), list(cmp[0].keys()) + ["mhc_used"])
        assertSameRows(rows, cmp, CMP_COLUMNS, scorecol="BigMHC_EL", tol=SCORE_TOL,
                       label="example2.prd", reflabel="example2.csv.cmp")
        self.assertEqual([r["mhc_used"] for r in rows], ["HLA-A*02:02"] * len(cmp))
        self.assertEqual(reportLines(proc.stderr), [])

    def test_testset_report_and_mhc_used(self):
        proc, rows, out = self.predict("testset", ["-i=" + self.testset, "-m=el", "-v=0"])
        self.assertEqual(proc.stdout, "")
        self.assertEqual(reportedInputs(proc.stderr), CONVERTED)
        self.assertIn("11 of 13 distinct", reportLines(proc.stderr)[0])
        lines = dict(zip(reportedInputs(proc.stderr), conversionLines(proc.stderr)))
        for m in BY_EDIT_DISTANCE:
            self.assertTrue(lines[m].startswith("MHC name conversion WARNING:"), m)
        for m in CONVERTED:
            if m not in BY_EDIT_DISTANCE:
                self.assertFalse(lines[m].startswith("MHC name conversion WARNING:"), m)
        self.assertIn("rows=2", lines["A0201"])
        self.assertIn("rows=2", lines["XYZ123"])
        self.assertIn("used='HLA-A*02:104' reason='truncation (x=1)'", lines["HLA-A*02:1040"])
        self.assertIn("used='HLA-B*13:101' reason='edit distance (d=3)'", lines["HLA-DRB1*01:01"])

        self.assertEqual(list(rows[0].keys()), ["mhc", "pep", "tgt", "len", "BigMHC_EL", "mhc_used"])
        self.assertEqual([(r["mhc"], r["pep"]) for r in rows], [(m, p) for m, p, _ in TESTSET])
        self.assertEqual([r["mhc_used"] for r in rows], [u for _, _, u in TESTSET])
        # rows that resolve to the same allele and peptide get the same score
        score = [r["BigMHC_EL"] for r in rows]
        self.assertEqual(len(set(score[i] for i in [0, 1, 2, 3, 7, 8])), 1)
        self.assertEqual(len(set(score[i] for i in [11, 12])), 1)
        # HLA-A*02:150 has the same pseudosequence as HLA-A*02:01
        self.assertEqual(len(set(score[i] for i in [9, 10, 13])), 1)
        self.assertNotEqual(score[0], score[4])

    def test_four_workers_give_the_same_file_and_report(self):
        """
        The run with four DataLoader workers must really use four (the
        DataLoader's num_workers is recorded), and its prediction file and
        conversion report must equal those of the single-worker run.
        """
        args = ["-i=" + self.testset, "-m=el", "-v=0"]
        proc1, rows1, out1, workers1 = self.predictRecordingWorkers("testset_j1", args, jobs=1)
        proc4, rows4, out4, workers4 = self.predictRecordingWorkers("testset_j4", args, jobs=4)
        self.assertEqual(workers1, [1])
        self.assertEqual(workers4, [4])
        with open(out1, "rb") as f1, open(out4, "rb") as f4:
            self.assertEqual(f1.read(), f4.read())
        self.assertEqual([r["mhc_used"] for r in rows4], [u for _, _, u in TESTSET])
        self.assertEqual(reportedInputs(proc4.stderr), CONVERTED)
        self.assertEqual(reportLines(proc4.stderr), reportLines(proc1.stderr))
        self.assertEqual(proc4.stderr.count("input='XYZ123'"), 1)

    def test_repeated_options_are_refused(self):
        """
        Guard against a caller option being silently overridden by a
        default (argparse keeps the last occurrence of an option).
        """
        with self.assertRaises(ValueError):
            predictArgs(["-i=" + self.testset, "-m=el", "-j=4"], os.path.join(self.tmp, "x.prd"), jobs=1)
        with self.assertRaises(ValueError):
            runScript("predict.py", ["-i=" + self.testset, "-m=el", "-j=4", "-d=cpu", "-j=1"])

    def test_fixed_shorthand_allele_reports_the_allele_used(self):
        proc, rows, _ = self.predict(
            "shorthand", ["-i=" + EXAMPLE2, "-m=el", "-a=A*02:02", "-p=0", "-c=0"])
        self.assertEqual([r["mhc"] for r in rows], ["A*02:02", "A*02:02"])
        self.assertEqual([r["mhc_used"] for r in rows], ["HLA-A*02:01", "HLA-A*02:01"])
        self.assertEqual(reportedInputs(proc.stderr), ["A*02:02"])
        self.assertIn("used='HLA-A*02:01' reason='HLA prefix + truncation (x=1)' rows=2",
                      conversionLines(proc.stderr)[0])

    def test_attention_columns_precede_mhc_used(self):
        proc, rows, _ = self.predict(
            "saveatt", ["-i=" + EXAMPLE2, "-m=el", "-a=HLA-A*02:02", "-p=0", "-c=0", "-z=1"])
        cols = list(rows[0].keys())
        self.assertEqual(cols[:5], ["mhc", "pep", "tgt", "len", "BigMHC_EL"])
        self.assertEqual(cols[5:35], ["att_{}".format(x) for x in range(30)])
        self.assertEqual(cols[35:], ["mhc_used"])

    def test_immunogenicity_model(self):
        proc, rows, _ = self.predict("example1_im", ["-i=" + EXAMPLE1, "-m=im", "-t=2"])
        self.assertEqual(list(rows[0].keys()), ["mhc", "pep", "tgt", "len", "BigMHC_IM", "mhc_used"])
        self.assertEqual([r["mhc_used"] for r in rows], [r["mhc"] for r in rows])
        self.assertEqual(reportLines(proc.stderr), [])

    def test_transfer_learning_reports_once(self):
        outdir = os.path.join(self.tmp, "tl")
        proc = runScript("train.py", [
            DEVICE, "-j=1",
            "-i=" + self.testset_tgt, "-m=" + os.path.join(testutil.MODELS, "bat512"),
            "-t=2", "-e=1", "-o=" + outdir])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("ep 1 loss:", proc.stdout)
        self.assertEqual(reportedInputs(proc.stderr), CONVERTED)
        self.assertEqual(len([f for f in os.listdir(os.path.join(outdir, "ep1"))
                              if f.endswith(".lyr")]), 4)


if __name__ == "__main__":
    unittest.main()
