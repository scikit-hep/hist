from hist import Hist, axis


def test_basic_usage():
    h = Hist(axis.Regular(10, 0, 1))

    h.fill([0.35, 0.35, 0.45])

    assert h[2] == 0
    assert h[3] == 2
    assert h[4] == 1
    assert h[5] == 0
