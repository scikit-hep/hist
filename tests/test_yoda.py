from __future__ import annotations
import numpy as np
import pytest

import hist
from hist import Hist, axis, storage

def test_read_yoda_str_1d():

    h1d = {
        "/some_h1d": Hist(hist.axis.Variable([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]), name="Histogram 1D")
    }
    
    h1d["/some_h1d"].fill(np.linspace(1,10,100))
    yoda_file1D = to_yoda_str(h1d)
    yoda_data = read_yoda_str(yoda_file1D)
    
    expected_results = {}

    for path, (class_name, header, body) in yoda_data.items():
        expected_results[path] = (class_name, header, body.strip())

    for path, (class_name, header, body) in yoda_data.items():

        if path in expected_results:
            expected_class, expected_header, expected_body = expected_results[path]
            assert class_name == expected_class
            assert body == expected_body
            assert header == expected_header

def test_read_yoda_str_2d():

    h2d = {
        "/some_h2d": Hist(
            hist.axis.Variable([1.0, 2.0, 3.0, 4.0, 5.0]),
            hist.axis.Variable([6.0, 7.0, 8.0, 9.0, 10.0]),
            name="Histogram 2D")
    }

    x_data = np.random.uniform(1, 5, 100)  
    y_data = np.random.uniform(6, 10, 100)  

    h2d["/some_h2d"].fill(x_data, y_data)
    yoda_file2D = to_yoda_str(h2d)

    yoda_data2D = read_yoda_str(yoda_file2D)

    expected_results = {}

    for path, (class_name, header, body) in yoda_data2D.items():
        expected_results[path] = (class_name, header, body.strip())

    for path, (class_name, header, body) in yoda_data2D.items():

        if path in expected_results:
            expected_class, expected_header, expected_body = expected_results[path]
            assert class_name == expected_class
            assert body == expected_body
            assert header == expected_header