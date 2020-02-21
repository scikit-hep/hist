from hist import axis, NamedHist


def test_basic_usage():
    h = NamedHist(
        axis.Regular(10, 0, 1, title="x")
    )  # NamedHist should require axis.Regular to have a name set

    h.fill([0.35, 0.35, 0.45])  # Fill should be keyword only, with the names

    assert h[2] == 0
    assert h[3] == 2
    assert h[4] == 1
    assert h[5] == 0

    # assert h[{'x':2}] # Should work
