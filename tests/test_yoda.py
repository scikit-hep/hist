from __future__ import annotations

import ctypes
import math

import numpy as np
import pytest


import hist
from hist import Hist

def test_read_yoda_str_1d():
    
    yoda_data = {
        "/some_h1d": (
            "YODA_HISTO1D_V2", 
            "some: stuff\n---\n",
            "# xlow\txhigh\tsumw\tsumw2\tsumwx\tsumwx2\tnumEntries\n"
            "1.000000e+00\t2.000000e+00\t1.000000e+00\t1.000000e+00\t1.500000e+00\t2.250000e+00\t1.000000e+02\n"
            "END"
        ),
    
    }

    for path, (class_name, header, body) in yoda_data.items():
        yoda_result = yoda_data.read_yoda_str(f"BEGIN {class_name} {path}\nPath: {path}\nTitle: Histogram 1D\nType: Histo1D\n{header}{body}\nEND {class_name}\n")

        expected_class, expected_header, expected_body = yoda_data[path]
        assert yoda_result[path] == (expected_class, expected_header, expected_body.strip())

def test_read_yoda_str_2d():

    yoda_data = {
        "/some_h2d": (
            "YODA_HISTO2D_V2", 
            "some: stuff\n---\n", 
            "# xlow\txhigh\tylow\tyhigh\tsumw\tsumw2\tsumwx\tsumwx2\tsumwy\tsumwy2\tsumwxy\tnumEntries\n"
            "1.000000e+00\t2.000000e+00\t6.000000e+00\t7.000000e+00\t1.000000e+00\t1.000000e+00\t1.500000e+00\t2.250000e+00\t6.500000e+00\t4.225000e+01\t9.750000e+00\t1.000000e+02\n"
            "END" 
        ),
    }

    for path, (class_name, header, body) in yoda_data.items():
        
        yoda_result = yoda_data.read_yoda_str(f"BEGIN {class_name} {path}\nPath: {path}\nTitle: Histogram 2D\nType: Histo2D\n{header}{body}\nEND {class_name}\n")

        expected_class, expected_header, expected_body = yoda_data[path]
        assert yoda_result[path] == (expected_class, expected_header, expected_body.strip())
