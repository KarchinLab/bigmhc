# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------
BigMHC Academic License

tests/test_predict_e2e.py
------------------------------------------------------------------------

End-to-end runs of predict.py and train.py on the CPU with the models of
the repository (skipped if they are absent). Each run loads the seven
models, so this module takes about a minute.

Checks: the README examples still reproduce their .cmp files (scores to
float32 rounding), the output keeps its original columns and gains
mhc_used last (also after the att_* columns of -z), the stderr report is
printed once (with -v=0 and with several DataLoader workers) and names the
alleles established on the unmodified code, both input modes work, and
rows that resolve to the same allele receive the same score.
"""

import os
import shutil
import tempfile
import unittest

import testutil
from testutil import conversionLines, reportLines, reportedInputs, readCSV, runScript

FIXED = ["-d=cpu", "-j=1"]

# work/inputs/testset_alleles.csv of the change's baseline: 17 rows,
# 13 distinct spellings, 8/9/10-mers, duplicate rows (1/8, 3/9, 7/17)
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


def scoresEqualWithin(a, b, col, tol):
    return all([abs(float(x[col]) - float(y[col])) <= tol for x, y in zip(a, b)])


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

    def predict(self, name, args):
        out = os.path.join(self.tmp, name + ".prd")
        proc = runScript("predict.py", list(args) + FIXED + ["-o=" + out])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(os.path.exists(out))
        return proc, readCSV(out), out

    def test_example1_reproduces_cmp_and_is_silent(self):
        proc, rows, _ = self.predict("example1", ["-i=" + EXAMPLE1, "-m=el", "-t=2"])
        cmp = readCSV(EXAMPLE1 + ".cmp")
        self.assertEqual(list(rows[0].keys()), list(cmp[0].keys()) + ["mhc_used"])
        self.assertEqual(len(rows), len(cmp))
        for r, c in zip(rows, cmp):
            for col in ["mhc", "pep", "tgt", "len"]:
                self.assertEqual(r[col], c[col])
            self.assertEqual(r["mhc_used"], r["mhc"])
        self.assertTrue(scoresEqualWithin(rows, cmp, "BigMHC_EL", 1e-5))
        self.assertEqual(reportLines(proc.stderr), [])

    def test_example2_fixed_allele_reproduces_cmp_and_is_silent(self):
        proc, rows, _ = self.predict(
            "example2", ["-i=" + EXAMPLE2, "-m=el", "-a=HLA-A*02:02", "-p=0", "-c=0"])
        cmp = readCSV(EXAMPLE2 + ".cmp")
        self.assertEqual(list(rows[0].keys()), list(cmp[0].keys()) + ["mhc_used"])
        for r, c in zip(rows, cmp):
            self.assertEqual((r["mhc"], r["pep"], r["len"]), (c["mhc"], c["pep"], c["len"]))
            self.assertEqual(r["mhc_used"], "HLA-A*02:02")
        self.assertTrue(scoresEqualWithin(rows, cmp, "BigMHC_EL", 1e-5))
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
        proc1, rows1, out1 = self.predict("testset_j1", ["-i=" + self.testset, "-m=el", "-v=0"])
        proc4, rows4, out4 = self.predict("testset_j4", ["-i=" + self.testset, "-m=el", "-v=0", "-j=4"])
        with open(out1, "rb") as f1, open(out4, "rb") as f4:
            self.assertEqual(f1.read(), f4.read())
        self.assertEqual(reportLines(proc4.stderr), reportLines(proc1.stderr))
        self.assertEqual(proc4.stderr.count("input='XYZ123'"), 1)

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
            "-i=" + self.testset_tgt, "-m=" + os.path.join(testutil.MODELS, "bat512"),
            "-t=2", "-e=1", "-o=" + outdir] + FIXED)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("ep 1 loss:", proc.stdout)
        self.assertEqual(reportedInputs(proc.stderr), CONVERTED)
        self.assertEqual(len([f for f in os.listdir(os.path.join(outdir, "ep1"))
                              if f.endswith(".lyr")]), 4)


if __name__ == "__main__":
    unittest.main()
