from __future__ import annotations

import boost_histogram as bh

import hist

# Test case 1: Creating from axes with name (should work)
a = hist.axis.Regular(10, 0, 10, name="r")
h1 = hist.Hist(a, name="hello")
print("Test 1 - Name from axes:", h1.name)  # Should print "hello"

# Test case 2: Creating from bh.Histogram with name (was broken)
bh_hist = bh.Histogram(a)
h2 = hist.Hist(bh_hist, name="world")
print("Test 2 - Name from bh.Histogram:", h2.name)  # Should print "world"
