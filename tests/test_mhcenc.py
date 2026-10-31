# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------
BigMHC Academic License

tests/test_mhcenc.py
------------------------------------------------------------------------

Name resolution layer: MHCEncoder.resolveMHC must report the allele that
MHCEncoder.nearestMHC selects, the selection must be the one made by the
original implementation (reference oracle in testutil), the encoding used
must be that allele's, cold and warm caches must agree, and the recorded
steps must describe the path actually taken.

The fast cases use a synthetic pseudosequence file; the cases on the real
data/pseudoseqs.csv check all 18,929 names and a fixed set of inputs whose
resolutions were established on the unmodified code.
"""

import os
import random
import shutil
import tempfile
import unittest

import torch

import testutil
from testutil import SYNTHETIC_NAMES, levenshtein, referenceNearest

from mhcuid import mhcuid
from mhcenc import MHCEncoder, MHCResolution


def _reasons(steps):
    return " + ".join(steps) if steps else "exact"


class SyntheticTableTests(unittest.TestCase):
    """
    Rules observed one by one on the synthetic pseudosequence file.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bigmhc_test_")
        cls.path = testutil.writeSyntheticPseudoseqs(
            os.path.join(cls.tmp, "pseudoseqs.csv"))
        cls.names = testutil.syntheticNames()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def encoder(self):
        return MHCEncoder.read(self.path)

    def test_read_keeps_table_names_in_file_order(self):
        enc = self.encoder()
        self.assertEqual(list(enc.encs.keys()),
                         [mhcuid(n) for n in SYNTHETIC_NAMES])
        self.assertEqual(list(enc.names.keys()), list(enc.encs.keys()))
        self.assertEqual(enc.names, self.names)
        self.assertEqual(enc.cols, testutil.SYNTHETIC_COLS)
        self.assertEqual(enc.enclen, len(testutil.SYNTHETIC_COLS))

    def test_names_default_to_keys_without_a_file(self):
        enc = self.encoder()
        bare = MHCEncoder(encs=enc.encs, cols=enc.cols)
        res = bare.resolveMHC("hla-a*02:01")
        self.assertEqual(res.used, "HLAA0201")
        self.assertEqual(res.steps, ["normalization"])

    def test_exact_name_is_not_a_conversion(self):
        res = self.encoder().resolveMHC("HLA-A*02:01")
        self.assertIsInstance(res, MHCResolution)
        self.assertEqual(res.mhc, "HLA-A*02:01")
        self.assertEqual(res.key, "HLAA0201")
        self.assertEqual(res.usedkey, "HLAA0201")
        self.assertEqual(res.used, "HLA-A*02:01")
        self.assertEqual(res.steps, [])
        self.assertEqual(res.reason, "exact")
        self.assertFalse(res.converted)
        self.assertFalse(res.byEditDistance)

    def test_normalization_only(self):
        enc = self.encoder()
        for raw in ["hla-a*02:01", "HLAA0201", "HLA-A02:01", "HLA-A*02:01 "]:
            res = enc.resolveMHC(raw)
            self.assertEqual(res.used, "HLA-A*02:01", raw)
            self.assertEqual(res.steps, ["normalization"], raw)
            self.assertTrue(res.converted, raw)
            self.assertFalse(res.byEditDistance, raw)

    def test_hla_prefix_is_followed_by_truncation_not_exact_lookup(self):
        """
        After "HLA" is prepended no exact lookup is made, so the first
        table entry sharing all but the last char wins: A0201 happens to
        reach HLA-A*02:01, but A0202 reaches HLA-A*02:01 as well even
        though HLA-A*02:02 is in the table. Preserved on purpose.
        """
        enc = self.encoder()
        expected = {
            "A0201":   "HLA-A*02:01",
            "A*02:01": "HLA-A*02:01",
            "A0202":   "HLA-A*02:01",
            "A*02:02": "HLA-A*02:01",
            "A0211":   "HLA-A*02:10",
            "B2705":   "HLA-B*27:01",
            "B*27:05": "HLA-B*27:01",
            "C0702":   "HLA-C*07:01",
            "B0702":   "HLA-B*07:02",
        }
        for raw, used in expected.items():
            res = enc.resolveMHC(raw)
            self.assertEqual(res.used, used, raw)
            self.assertEqual(res.steps, ["HLA prefix", "truncation (x=1)"], raw)
            self.assertEqual(res.reason, "HLA prefix + truncation (x=1)", raw)
            self.assertFalse(res.byEditDistance, raw)

    def test_truncation(self):
        enc = self.encoder()
        expected = {
            "HLA-A*02:01:01":     ("HLA-A*02:01", 2),
            "HLA-A*02:01:01:02L": ("HLA-A*02:01", 5),
            "HLA-A*02:04":        ("HLA-A*02:01", 1),   # absent allele
            "HLA-A*02:0":         ("HLA-A*02:01", 1),   # 5 candidates tie
            "HLA-B*27:0":         ("HLA-B*27:01", 1),
        }
        for raw, (used, x) in expected.items():
            res = enc.resolveMHC(raw)
            self.assertEqual(res.used, used, raw)
            self.assertEqual(res.steps, ["truncation (x={})".format(x)], raw)

    def test_edit_distance_with_tie_breaking(self):
        """
        When several table entries share the minimal Levenshtein distance
        the first one in file order is used, as in the original code.
        """
        enc = self.encoder()
        for raw in ["HLA-DRB1*01:01", "QQQ", "XYZ123"]:
            key = mhcuid(raw)
            dists = [(levenshtein(k, key), k) for k in enc.encs]
            dmin = min(d for d, _ in dists)
            ties = [k for d, k in dists if d == dmin]
            res = enc.resolveMHC(raw)
            self.assertEqual(res.usedkey, ties[0], raw)
            self.assertEqual(res.used, self.names[ties[0]], raw)
            self.assertEqual(res.steps, ["edit distance (d={})".format(dmin)], raw)
            self.assertTrue(res.byEditDistance, raw)
        self.assertGreaterEqual(
            len([k for k in enc.encs if levenshtein(k, "QQQ") == 4]), 2,
            "QQQ was meant to tie between H2DB and H2KB")
        self.assertEqual(enc.resolveMHC("QQQ").used, "H2-Db")

    def test_hla_prefix_followed_by_edit_distance(self):
        enc = self.encoder()
        res = enc.resolveMHC("Z9")
        self.assertEqual(res.steps[0], "HLA prefix")
        self.assertTrue(res.steps[1].startswith("edit distance (d="))
        self.assertTrue(res.byEditDistance)
        self.assertEqual(res.usedkey, referenceNearest(enc.encs, "Z9"))

    def test_short_name_skips_truncation(self):
        enc = self.encoder()
        res = enc.resolveMHC("AB")
        self.assertEqual(len(res.steps), 1)
        self.assertTrue(res.steps[0].startswith("edit distance (d="))
        self.assertEqual(res.usedkey, referenceNearest(enc.encs, "AB"))

    def test_resolution_matches_reference_and_encoding(self):
        """
        For many generated spellings: the key chosen by nearestMHC is the
        one the original implementation chooses (cold cache, warm cache
        and a shared encoder), resolveMHC reports that key and the table
        name, and __getitem__ returns that key's encoding.
        """
        rng = random.Random(1)
        inputs = list(SYNTHETIC_NAMES)
        alphabet = "ABCDEFGHKLMNPQRSTUVWXYZ0123456789"
        for _ in range(600):
            s = rng.choice(SYNTHETIC_NAMES)
            op = rng.randrange(7)
            if op == 0:
                s = s.lower()
            elif op == 1:
                s = s.replace("HLA-", "")
            elif op == 2:
                s = s.replace("*", "").replace(":", "")
            elif op == 3:
                i = rng.randrange(len(s))
                s = s[:i] + s[i+1:]
            elif op == 4:
                i = rng.randrange(len(s))
                s = s[:i] + rng.choice(alphabet) + s[i+1:]
            elif op == 5:
                s = s + ":" + "".join(rng.choice("0123456789") for _ in range(2))
            else:
                s = "".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 12)))
            inputs.append(s)
        shared = self.encoder()
        for raw in inputs:
            key = mhcuid(raw)
            if not key:
                continue
            expected = referenceNearest(shared.encs, key)
            cold = self.encoder()
            self.assertEqual(cold.nearestMHC(key), expected, raw)
            self.assertEqual(cold.nearestMHC(key), expected, raw)       # warm
            self.assertEqual(shared.nearestMHC(key), expected, raw)     # shared
            res = shared.resolveMHC(raw)
            self.assertEqual(res.usedkey, expected, raw)
            self.assertEqual(res.used, self.names[expected], raw)
            self.assertEqual(res.key, key, raw)
            self.assertEqual(res.converted, raw != self.names[expected], raw)
            self.assertTrue(torch.equal(shared[raw], shared.encs[expected]), raw)
            self.assertTrue(torch.equal(cold[raw], shared.encs[expected]), raw)

    def test_resolve_leaves_subcache_untouched(self):
        enc = self.encoder()
        for raw in ["A0201", "HLA-A*02:01:01", "XYZ123", "hla-a*02:01"]:
            enc.resolveMHC(raw)
        self.assertEqual(enc.subcache, {})

    def test_subcache_behaves_as_before(self):
        """
        Substitutes are cached under the searched (possibly prefixed) key,
        exactly as the original code did; exact keys are never cached.
        """
        enc = self.encoder()
        enc.nearestMHC("HLAA020101")
        self.assertEqual(enc.subcache, {"HLAA020101": "HLAA0201"})
        enc.nearestMHC("A0201")
        self.assertEqual(enc.subcache["HLAA0201"], "HLAA0201")
        self.assertNotIn("A0201", enc.subcache)
        enc.nearestMHC("HLAA0201")
        self.assertEqual(len(enc.subcache), 2)

    def test_setitem_records_the_given_name(self):
        enc = self.encoder()
        enc["HLA-Z*99:99"] = torch.zeros(enc.enclen, dtype=torch.int8)
        self.assertEqual(enc.names["HLAZ9999"], "HLA-Z*99:99")
        res = enc.resolveMHC("hla-z*99:99")
        self.assertEqual(res.used, "HLA-Z*99:99")
        self.assertEqual(res.steps, ["normalization"])
        del enc["HLA-Z*99:99"]
        self.assertNotIn("HLAZ9999", enc.encs)
        self.assertNotIn("HLAZ9999", enc.names)

    def test_repr(self):
        res = self.encoder().resolveMHC("A0201")
        self.assertEqual(
            repr(res),
            "MHCResolution(mhc='A0201', used='HLA-A*02:01',"
            " reason='HLA prefix + truncation (x=1)')")


# Expected resolutions for data/pseudoseqs.csv: the entry selected by the
# original nearestMHC() of KarchinLab/bigmhc commit 199e890 (the base of
# this change) for each input, and the rule(s) that search applied to
# reach it. The original search is kept verbatim as referenceNearest in
# testutil.
REAL_TABLE_EXPECTED = {
    "HLA-A*02:01":        ("HLA-A*02:01",  "exact"),
    "hla-a*02:01":        ("HLA-A*02:01",  "normalization"),
    "HLA-A02:01":         ("HLA-A*02:01",  "normalization"),
    "HLAA0201":           ("HLA-A*02:01",  "normalization"),
    "A0201":              ("HLA-A*02:01",  "HLA prefix + truncation (x=1)"),
    "A*02:01":            ("HLA-A*02:01",  "HLA prefix + truncation (x=1)"),
    "A0202":              ("HLA-A*02:01",  "HLA prefix + truncation (x=1)"),
    "A*02:02":            ("HLA-A*02:01",  "HLA prefix + truncation (x=1)"),
    "A0211":              ("HLA-A*02:10",  "HLA prefix + truncation (x=1)"),
    "A2402":              ("HLA-A*24:02",  "HLA prefix + truncation (x=1)"),
    "B2705":              ("HLA-B*27:01",  "HLA prefix + truncation (x=1)"),
    "B*27:05":            ("HLA-B*27:01",  "HLA prefix + truncation (x=1)"),
    "B0702":              ("HLA-B*07:02",  "HLA prefix + truncation (x=1)"),
    "C0702":              ("HLA-C*07:01",  "HLA prefix + truncation (x=1)"),
    "HLA-A*02:01:01":     ("HLA-A*02:01",  "truncation (x=2)"),
    "HLA-A*02:01:01:02L": ("HLA-A*02:01",  "truncation (x=5)"),
    "HLA-A*02:1040":      ("HLA-A*02:104", "truncation (x=1)"),
    "HLA-A*02:15N":       ("HLA-A*02:150", "truncation (x=1)"),
    "HLA-DRB1*01:01":     ("HLA-B*13:101", "edit distance (d=3)"),
    "hla-drb1*01:01":     ("HLA-B*13:101", "edit distance (d=3)"),
    "XYZ123":             ("HLA-A*01:23",  "edit distance (d=5)"),
    "A*02:01:01:02L":     ("Atfu-B*01:02", "edit distance (d=5)"),
    "Z9":                 ("H2-Db",        "HLA prefix + edit distance (d=4)"),
    "HLA-A*02:02":        ("HLA-A*02:02",  "exact"),
    "HLA-A*11:01":        ("HLA-A*11:01",  "exact"),
    "HLA-B*07:02":        ("HLA-B*07:02",  "exact"),
    "HLA-B*08:01":        ("HLA-B*08:01",  "exact"),
    "HLA-B*27:01":        ("HLA-B*27:01",  "exact"),
    "HLA-B*27:05":        ("HLA-B*27:05",  "exact"),
    "HLA-C*03:03":        ("HLA-C*03:03",  "exact"),
    "HLA-C*05:01":        ("HLA-C*05:01",  "exact"),
    "HLA-C*12:03":        ("HLA-C*12:03",  "exact"),
}


@unittest.skipUnless(os.path.exists(testutil.PSEUDOSEQS),
                     "data/pseudoseqs.csv not found")
class RealTableTests(unittest.TestCase):
    """
    The real pseudosequence file: every table name resolves to itself and
    the fixed input set resolves as it did before the change.
    """

    @classmethod
    def setUpClass(cls):
        cls.enc = MHCEncoder.read(testutil.PSEUDOSEQS)

    def test_table_size_and_names(self):
        self.assertEqual(len(self.enc), 18929)
        self.assertEqual(len(self.enc.names), 18929)
        self.assertEqual(self.enc.enclen, 414)
        self.assertEqual(self.enc.names["HLAA0201"], "HLA-A*02:01")

    def test_every_table_name_resolves_to_itself(self):
        for key, name in self.enc.names.items():
            res = self.enc.resolveMHC(name)
            self.assertEqual(res.key, key)
            self.assertEqual(res.usedkey, key)
            self.assertEqual(res.used, name)
            self.assertFalse(res.converted, name)
            self.assertEqual(self.enc.nearestMHC(key), key)
        # exact keys are never cached
        self.assertFalse(set(self.enc.names) & set(self.enc.subcache))

    def test_fixed_inputs_resolve_as_before(self):
        """
        used name and reason as established on the unmodified code; key
        equal to the reference oracle; encoding equal to that key's;
        first (cold) and second (warm) call agree, as does a second
        encoder visiting the inputs in reverse order.
        """
        reverse = MHCEncoder.read(testutil.PSEUDOSEQS)
        for raw in reversed(list(REAL_TABLE_EXPECTED)):
            reverse.nearestMHC(mhcuid(raw))
        for raw, (used, reason) in REAL_TABLE_EXPECTED.items():
            key = mhcuid(raw)
            res = self.enc.resolveMHC(raw)
            self.assertEqual(res.used, used, raw)
            self.assertEqual(res.reason, reason, raw)
            self.assertEqual(res.usedkey, referenceNearest(self.enc.encs, key), raw)
            self.assertEqual(self.enc.nearestMHC(key), res.usedkey, raw)   # cold
            self.assertEqual(self.enc.nearestMHC(key), res.usedkey, raw)   # warm
            self.assertEqual(reverse.nearestMHC(key), res.usedkey, raw)
            self.assertTrue(torch.equal(self.enc[raw], self.enc.encs[res.usedkey]), raw)
            self.assertEqual(res.byEditDistance, "edit distance" in reason, raw)
            self.assertEqual(res.converted, reason != "exact", raw)


if __name__ == "__main__":
    unittest.main()
