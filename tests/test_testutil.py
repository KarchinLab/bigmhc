# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------
BigMHC Academic License

tests/test_testutil.py
------------------------------------------------------------------------

The checks the other test modules rely on must themselves fail when they
should: assertSameRows must reject a missing or extra row (not just
compare the rows that zip() happens to pair), and the repeated-option
guard must reject an argument list in which a default overrides a
caller's option. No models are needed.
"""

import os
import unittest

import testutil
from testutil import assertSameRows, checkNoRepeatedOptions, optionNames, readCSV, runScript

import test_predict_e2e

CMP2 = os.path.join(testutil.DATA, "example2.csv.cmp")
CMP1 = os.path.join(testutil.DATA, "example1.csv.cmp")
COLS = ["mhc", "pep", "tgt", "len"]


class AssertSameRowsTests(unittest.TestCase):

    def setUp(self):
        self.cmp2 = readCSV(CMP2)
        self.cmp1 = readCSV(CMP1)
        self.assertEqual(len(self.cmp2), 2)
        self.assertEqual(len(self.cmp1), 9)

    def test_identical_rows_pass(self):
        assertSameRows(self.cmp2, self.cmp2, COLS, scorecol="BigMHC_EL", tol=0.0)
        assertSameRows(self.cmp1, self.cmp1, COLS, scorecol="BigMHC_EL", tol=0.0)

    def test_one_row_missing_fails(self):
        with self.assertRaises(AssertionError) as cm:
            assertSameRows(self.cmp2[:-1], self.cmp2, COLS, scorecol="BigMHC_EL", tol=1e-5,
                           label="output", reflabel="example2.csv.cmp")
        self.assertIn("output has 1 rows but example2.csv.cmp has 2", str(cm.exception))
        with self.assertRaises(AssertionError):
            assertSameRows(self.cmp1[1:], self.cmp1, COLS, scorecol="BigMHC_EL", tol=1e-5)

    def test_one_row_extra_fails(self):
        with self.assertRaises(AssertionError):
            assertSameRows(self.cmp2 + self.cmp2[:1], self.cmp2, COLS)

    def test_empty_output_fails(self):
        with self.assertRaises(AssertionError):
            assertSameRows([], self.cmp2, COLS)

    def test_changed_text_column_fails(self):
        rows = [dict(r) for r in self.cmp2]
        rows[1]["pep"] = rows[1]["pep"][:-1] + "A"
        with self.assertRaises(AssertionError) as cm:
            assertSameRows(rows, self.cmp2, COLS)
        self.assertIn("row 1: pep differs", str(cm.exception))

    def test_score_tolerance(self):
        rows = [dict(r) for r in self.cmp2]
        rows[0]["BigMHC_EL"] = str(float(rows[0]["BigMHC_EL"]) + 2e-7)
        assertSameRows(rows, self.cmp2, COLS, scorecol="BigMHC_EL", tol=1e-5)
        with self.assertRaises(AssertionError):
            assertSameRows(rows, self.cmp2, COLS, scorecol="BigMHC_EL", tol=1e-8)
        rows[0]["BigMHC_EL"] = "nan"
        with self.assertRaises(AssertionError):
            assertSameRows(rows, self.cmp2, COLS, scorecol="BigMHC_EL", tol=1.0)


class RepeatedOptionGuardTests(unittest.TestCase):

    def test_option_names(self):
        self.assertEqual(optionNames(["-i=a.csv", "-m=el", "-v=0", "-j=4", "x"]),
                         ["-i", "-m", "-v", "-j"])

    def test_unique_options_pass(self):
        checkNoRepeatedOptions(["-i=a.csv", "-m=el", "-d=cpu", "-j=1", "-o=out.prd"])

    def test_default_appended_after_caller_option_is_refused(self):
        """
        The construction the e2e suite used before: defaults appended
        after the caller's arguments, so -j=4 was overridden by -j=1.
        """
        caller = ["-i=a.csv", "-m=el", "-v=0", "-j=4"]
        with self.assertRaises(ValueError) as cm:
            checkNoRepeatedOptions(caller + ["-d=cpu", "-j=1"])
        self.assertIn("-j", str(cm.exception))

    def test_runScript_refuses_before_running(self):
        with self.assertRaises(ValueError):
            runScript("predict.py", ["-i=a.csv", "-j=4", "-j=1"])

    def test_predictArgs_puts_jobs_once(self):
        args = test_predict_e2e.predictArgs(["-i=a.csv", "-m=el", "-v=0"], "out.prd", jobs=4)
        self.assertEqual(args.count("-j=4"), 1)
        self.assertNotIn("-j=1", args)
        self.assertEqual(optionNames(args).count("-j"), 1)
        self.assertEqual(args[0], test_predict_e2e.DEVICE)
        with self.assertRaises(ValueError):
            test_predict_e2e.predictArgs(["-i=a.csv", "-m=el", "-j=4"], "out.prd", jobs=1)


if __name__ == "__main__":
    unittest.main()
