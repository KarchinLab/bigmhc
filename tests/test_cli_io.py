# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------
BigMHC Academic License

tests/test_cli_io.py
------------------------------------------------------------------------

Input/output layer around cli._resolveAlleles and cli._loadData, using
the synthetic pseudosequence file (no models needed):
  * the stderr report lists each converted input once with its row count,
    is printed with verbose off, and is silent for exact names
  * the fixed -a allele and the allele column behave the same
  * mhc_used stays with its row through batching (mixed peptide lengths,
    duplicate rows, small batches) and the encoding each row receives is
    the one of the allele reported in mhc_used
  * DataLoader workers (spawned processes) add nothing to stderr
  * train.py shares the entry point and gets the same single report
"""

import argparse
import io
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

from contextlib import redirect_stderr, redirect_stdout

import pandas as pd
import torch

import testutil
from testutil import conversionLines, reportLines, reportedInputs

import cli

from mhcuid import mhcuid
from mhcenc import MHCEncoder


# 16 rows, 12 distinct spellings, 3 peptide lengths, duplicate rows
TESTSET = [
    ("HLA-A*02:01",    "GILGFVFTL"),
    ("hla-a*02:01",    "GILGFVFTL"),
    ("A0201",          "GILGFVFTL"),
    ("HLA-A*02:01:01", "GILGFVFTL"),
    ("HLA-A*02:04",    "GILGFVFTL"),
    ("HLA-DRB1*01:01", "GILGFVFTL"),
    ("XYZ123",         "GILGFVFTL"),
    ("HLA-A*02:01",    "GILGFVFTL"),
    ("A0201",          "GILGFVFTL"),
    ("HLA-A*02:01",    "FLPSDFFPSV"),
    ("A*02:01",        "FLPSDFFPSV"),
    ("HLA-A02:01",     "FAYLRDLL"),
    ("HLAA0201",       "FAYLRDLL"),
    ("HLA-B*07:02",    "GILGFVFTL"),
    ("hla-drb1*01:01", "FAYLRDLL"),
    ("XYZ123",         "FLPSDFFPSV"),
]

EXACT_INPUTS = ["HLA-A*02:01", "HLA-B*07:02"]


class CLIIOTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bigmhc_test_")
        cls.pseudoseqs = testutil.writeSyntheticPseudoseqs(
            os.path.join(cls.tmp, "pseudoseqs.csv"))
        cls.enc = MHCEncoder.read(cls.pseudoseqs)
        cls.testset = testutil.writeCSV(
            os.path.join(cls.tmp, "testset.csv"),
            TESTSET, header=["mhc", "pep"])
        cls.testset_tgt = testutil.writeCSV(
            os.path.join(cls.tmp, "testset_tgt.csv"),
            [(m, p, i % 2) for i, (m, p) in enumerate(TESTSET)],
            header=["mhc", "pep", "tgt"])
        cls.peptides = testutil.writeCSV(
            os.path.join(cls.tmp, "peptides.csv"),
            [(p,) for p in ["GILGFVFTL", "FLPSDFFPSV", "FAYLRDLL"]])
        cls.exact = testutil.writeCSV(
            os.path.join(cls.tmp, "exact.csv"),
            [("HLA-A*02:01", "GILGFVFTL"), ("HLA-B*07:02", "FAYLRDLL"),
             ("HLA-A*02:01", "FLPSDFFPSV")],
            header=["mhc", "pep"])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def args(self, **kw):
        """
        Namespace with the fields _loadData reads, as parseArgs sets them
        """
        a = argparse.Namespace(
            input=self.testset,
            allele=0,
            pepcol=1,
            tgtcol=None,
            hdrcnt=1,
            verbose=1,
            pseudoseqs=self.pseudoseqs,
            maxbat=1000,
            train=False,
            transferlearn=False,
            jobs=1,
            prefetch=2)
        for k, v in kw.items():
            setattr(a, k, v)
        return a

    def loadData(self, **kw):
        """
        Runs cli._loadData capturing stdout and stderr
        """
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            data = cli._loadData(self.args(**kw))
        return data, out.getvalue(), err.getvalue()

    def expectedUsed(self, mhcs):
        return [self.enc.resolveMHC(m).used for m in mhcs]

    # ------------------------------------------------------------------
    # report content

    def test_report_lists_each_converted_input_once(self):
        data, out, err = self.loadData()
        distinct = list(dict.fromkeys(m for m, _ in TESTSET))
        converted = [m for m in distinct if m not in EXACT_INPUTS]
        self.assertEqual(reportedInputs(err), converted)
        header = reportLines(err)[0]
        self.assertTrue(header.startswith("MHC name conversions: {} of {} distinct".format(
            len(converted), len(distinct))), header)
        self.assertIn("mhc_used", header)
        lines = conversionLines(err)
        self.assertEqual(len(lines), len(converted))
        for m in EXACT_INPUTS:
            self.assertNotIn("input='{}'".format(m), err)
        byinput = dict(zip(reportedInputs(err), lines))
        self.assertIn("rows=2", byinput["A0201"])
        self.assertIn("rows=2", byinput["XYZ123"])
        self.assertIn("rows=1", byinput["hla-a*02:01"])
        self.assertIn("used='HLA-A*02:01' reason='normalization'", byinput["hla-a*02:01"])
        self.assertIn("used='HLA-A*02:01' reason='HLA prefix + truncation (x=1)'", byinput["A0201"])
        self.assertIn("used='HLA-A*02:01' reason='truncation (x=2)'", byinput["HLA-A*02:01:01"])
        self.assertIn("used='HLA-A*02:01' reason='truncation (x=1)'", byinput["HLA-A*02:04"])
        for m in ["HLA-DRB1*01:01", "hla-drb1*01:01", "XYZ123"]:
            self.assertTrue(byinput[m].startswith("MHC name conversion WARNING:"), m)
            self.assertIn("reason='edit distance (d=", byinput[m])
            self.assertIn("does not imply any biological similarity", byinput[m])
        for m in converted:
            if m not in ["HLA-DRB1*01:01", "hla-drb1*01:01", "XYZ123"]:
                self.assertNotIn("WARNING", byinput[m], m)
        self.assertEqual(err.count("input='A0201'"), 1)

    def test_exact_names_are_silent_but_mhc_used_is_filled(self):
        data, out, err = self.loadData(input=self.exact)
        self.assertEqual(reportLines(err), [])
        df = data.dataset.df
        self.assertIn("mhc_used", df.columns)
        self.assertEqual(df["mhc_used"].tolist(), df["mhc"].tolist())

    def test_report_is_printed_with_verbose_off(self):
        data, out, err = self.loadData(verbose=0)
        self.assertEqual(out, "")
        self.assertEqual(len(conversionLines(err)), 10)
        data1, out1, err1 = self.loadData(verbose=1)
        self.assertIn("processing", out1)
        self.assertEqual(reportLines(err1), reportLines(err))

    def test_resolve_alleles_adds_only_mhc_used(self):
        from dataset import Dataset
        before = Dataset.readPMHCs(
            fp=self.testset, allele=0, pepcol=1, hdrcnt=1, verbose=False)
        after = cli._resolveAlleles(before.copy(), self.enc, file=io.StringIO())
        self.assertEqual(list(after.columns), ["mhc", "pep", "tgt", "len", "mhc_used"])
        self.assertEqual(after["mhc_used"].tolist(), self.expectedUsed(after["mhc"].tolist()))
        self.assertEqual(str(after["mhc_used"].dtype), str(after["mhc"].dtype))
        pd.testing.assert_frame_equal(after[before.columns], before)

    # ------------------------------------------------------------------
    # the two input modes

    def test_fixed_allele_and_allele_column_agree(self):
        data_a, _, err_a = self.loadData(
            input=self.peptides, allele="A*02:02", pepcol=0, hdrcnt=0)
        self.assertEqual(data_a.dataset.df["mhc"].tolist(), ["A*02:02"] * 3)
        self.assertEqual(data_a.dataset.df["mhc_used"].tolist(), ["HLA-A*02:01"] * 3)
        self.assertEqual(reportedInputs(err_a), ["A*02:02"])
        self.assertIn("rows=3", conversionLines(err_a)[0])
        self.assertIn("1 of 1 distinct", err_a)

        column = testutil.writeCSV(
            os.path.join(self.tmp, "column.csv"),
            [("A*02:02", p) for p in ["GILGFVFTL", "FLPSDFFPSV", "FAYLRDLL"]])
        data_c, _, err_c = self.loadData(input=column, allele=0, pepcol=1, hdrcnt=0)
        self.assertEqual(reportLines(err_c), reportLines(err_a))
        self.assertEqual(data_c.dataset.df["mhc_used"].tolist(),
                         data_a.dataset.df["mhc_used"].tolist())

    # ------------------------------------------------------------------
    # per-row alignment through batching

    def checkBatches(self, data):
        """
        For every batch: the raw rows carry the right mhc_used for their
        mhc, and the encoded MHC of each row is the encoding of the allele
        named in mhc_used. Returns the raw batches concatenated and sorted
        as predict.py does.
        """
        ds = data.dataset
        raws = list()
        for idx in range(len(ds)):
            raw = ds.getbat(idx=idx, enc=False)
            bat = ds.getbat(idx=idx, enc=True)
            self.assertEqual(len(raw), len(bat))
            self.assertEqual(raw["len"].nunique(), 1)
            for i in range(len(raw)):
                mhc = raw["mhc"].iloc[i]
                used = raw["mhc_used"].iloc[i]
                self.assertEqual(used, self.enc.resolveMHC(mhc).used)
                self.assertTrue(torch.equal(bat.mhc[i], self.enc.encs[mhcuid(used)]))
            raws.append(raw)
        return pd.concat(raws).sort_index()

    def test_mhc_used_stays_with_its_row(self):
        data, _, _ = self.loadData()
        ds = data.dataset
        self.assertEqual(len(ds), 3)        # one batch per peptide length
        preds = self.checkBatches(data)
        self.assertEqual(preds["mhc"].tolist(), [m for m, _ in TESTSET])
        self.assertEqual(preds["pep"].tolist(), [p for _, p in TESTSET])
        self.assertEqual(preds["mhc_used"].tolist(),
                         self.expectedUsed([m for m, _ in TESTSET]))

    def test_mhc_used_stays_with_its_row_in_small_batches(self):
        data, _, _ = self.loadData(maxbat=2)
        self.assertGreater(len(data.dataset), 3)
        preds = self.checkBatches(data)
        self.assertEqual(preds["mhc"].tolist(), [m for m, _ in TESTSET])
        self.assertEqual(preds["mhc_used"].tolist(),
                         self.expectedUsed([m for m, _ in TESTSET]))

    def test_encoded_batches_from_dataloader(self):
        """
        The batches delivered by the DataLoader (one worker) carry the
        encodings of the alleles named in mhc_used.
        """
        data, _, _ = self.loadData(jobs=1, maxbat=2)
        ds = data.dataset
        for idx, bat in enumerate(data):
            raw = ds.getbat(idx=idx, enc=False)
            for i in range(len(raw)):
                used = raw["mhc_used"].iloc[i]
                self.assertTrue(torch.equal(bat.mhc[i], self.enc.encs[mhcuid(used)]))

    # ------------------------------------------------------------------
    # processes

    DRIVER = textwrap.dedent("""
        import argparse, sys
        sys.path.insert(0, {src!r})
        import cli
        args = argparse.Namespace(
            input={input!r}, allele=0, pepcol=1, tgtcol=None, hdrcnt=1,
            verbose=0, pseudoseqs={pseudoseqs!r}, maxbat=2, train=False,
            transferlearn=False, jobs={jobs}, prefetch=2)
        data = cli._loadData(args)
        n = sum(1 for _ in data)
        print("batches", n)
        """)

    def runDriver(self, jobs):
        code = self.DRIVER.format(
            src=testutil.SRC, input=self.testset,
            pseudoseqs=self.pseudoseqs, jobs=jobs)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc

    def test_workers_do_not_repeat_the_report(self):
        lens = [len(p) for _, p in TESTSET]
        nbats = sum(-(-lens.count(l) // 2) for l in set(lens))   # maxbat=2
        one = self.runDriver(jobs=1)
        four = self.runDriver(jobs=4)
        self.assertEqual(one.stdout.strip(), "batches {}".format(nbats))
        self.assertEqual(four.stdout.strip(), "batches {}".format(nbats))
        self.assertEqual(len(conversionLines(one.stderr)), 10)
        self.assertEqual(reportLines(four.stderr), reportLines(one.stderr))
        self.assertEqual(four.stderr.count("input='XYZ123'"), 1)

    def test_training_entry_point_reports_once(self):
        data, out, err = self.loadData(
            input=self.testset_tgt, tgtcol=2, train=True,
            transferlearn=True, maxbat=32)
        _, _, err_predict = self.loadData()
        self.assertEqual(reportLines(err), reportLines(err_predict))
        self.assertEqual(len(conversionLines(err)), 10)
        df = data.dataset.df
        self.assertEqual(df["mhc_used"].tolist(), self.expectedUsed(df["mhc"].tolist()))


if __name__ == "__main__":
    unittest.main()
