from __future__ import annotations

import ctypes
import math

import numpy as np
import pytest


import hist
from hist import Hist
def test_read_yoda_str():
    h = """BEGIN YODA_HISTO1D_V2 /some_h1d
                    Path: /some_h1d
                    Title: title
                    Type: Histo1D
                    some: stuff
                    ---
                    # Mean: -1.826931e-02
                    # Area: 1.000000e+03
                    # ID\tID\tsumw\tsumw2\tsumwx\tsumwx2\tnumEntries
                    0.000000e+00	1.000000e+00	1.000000e+00	1.000000e+00	5.000000e-01	2.500000e-01	1.000000e+00
                    END YODA_HISTO1D_V2

                    BEGIN YODA_HISTO2D_V2 /some_h2d
                    Path: /some_h2d
                    Title: Histogram 2D
                    Type: Histo2D
                    some: stuff
                    ---
                    # Mean: (2.980000e+00, 7.980000e+00)
                    # Volume: 1.000000e+02
                    # ID\tID\tsumw\tsumw2\tsumwx\tsumwx2\tsumwy\tsumwy2\tsumwxy\tnumEntries
                    0.000000e+00	1.000000e+00	0.000000e+00	0.000000e+00	0.000000e+00	0.000000e+00	0.000000e+00	0.000000e+00	0.000000e+00	0.000000e+00
                    1.000000e+00	2.000000e+00	1.000000e+00	1.000000e+00	1.500000e+00	2.250000e+00	1.500000e+00	2.250000e+00	3.750000e+00	1.000000e+00
                    END YODA_HISTO2D_V2
                    """
    expected_output = {
        "/some_h1d": ("YODA_HISTO1D", "Path: /some_h1d\nTitle: Histogram 1D\nType: Histo1D\nsome: stuff\n---\n", "# ID\tID\tsumw\tsumw2\tsumwx\tsumwx2\tnumEntries\n0.000000e+00\t1.000000e+00\t1.000000e+00\t1.000000e+00\t5.000000e-01\t2.500000e-01\t1.000000e+00\nEND YODA_HISTO1D_V2"),
        "/some_h2d": ("YODA_HISTO2D", "Path: /some_h2d\nTitle: Histogram 2D\nType: Histo2D\nsome: stuff\n---\n", "# ID\tID\tsumw\tsumw2\tsumwx\tsumwx2\tsumwy\tsumwy2\tsumwxy\tnumEntries\n0.000000e+00\t1.000000e+00\t0.000000e+00\t0.000000e+00\t0.000000e+00\t0.000000e+00\t0.000000e+00\t0.000000e+00\t0.000000e+00\t0.000000e+00\n1.000000e+00\t2.000000e+00\t1.000000e+00\t1.000000e+00\t1.500000e+00\t2.250000e+00\t1.500000e+00\t2.250000e+00\t3.750000e+00\t1.000000e+00\nEND YODA_HISTO2D_V2")
    }


    result = h.read_yoda_str(h)
    
    assert result.keys() == expected_output.keys(), "Keys mismatch"
    for key in result.keys():
        assert result[key][0] == expected_output[key][0], f"Class name mismatch for {key}\nExpected: {expected_output[key][0]}\nGot: {result[key][0]}"
        assert result[key][1] == expected_output[key][1], f"Header mismatch for {key}\nExpected: {expected_output[key][1]}\nGot: {result[key][1]}"
        assert result[key][2] == expected_output[key][2], f"Body mismatch for {key}\nExpected: {expected_output[key][2]}\nGot: {result[key][2]}"
