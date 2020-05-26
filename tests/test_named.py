from hist import axis, NamedHist
import boost_histogram as bh


def test_basic_usage():
    """
        Test basic usage -- whether NamedHist are properly derived from\
        boost-histogram and whether it can be filled by names.
    """

    h = NamedHist(axis.Regular(10, 0, 1, name="x")).fill(x=[0.35, 0.35, 0.45])

    for idx in range(10):
        if idx == 3:
            assert h[idx] == h[{0: idx}] == h[{"x": idx}] == 2
        elif idx == 4:
            assert h[idx] == h[{0: idx}] == h[{"x": idx}] == 1
        else:
            assert h[idx] == h[{0: idx}] == h[{"x": idx}] == 0

    h = NamedHist(
        axis.Regular(10, 0, 1, name="x"),
        axis.Regular(10, 0, 1, name="y"),
        axis.Integer(0, 2, name="z"),
    ).fill(
        x=[0.35, 0.35, 0.35, 0.45, 0.55, 0.55, 0.55],
        y=[0.35, 0.35, 0.45, 0.45, 0.45, 0.45, 0.45],
        z=[0, 0, 1, 1, 1, 1, 1],
    )

    z_one_only = h[{"z": bh.loc(1)}]
    for idx_x in range(0, 10):
        for idx_y in range(0, 10):
            if idx_x == 3 and idx_y == 4:
                assert (
                    z_one_only[idx_x, idx_y]
                    == z_one_only[{"x": idx_x, "y": idx_y}]
                    == 1
                )
            elif idx_x == 4 and idx_y == 4:
                assert (
                    z_one_only[idx_x, idx_y]
                    == z_one_only[{"x": idx_x, "y": idx_y}]
                    == 1
                )
            elif idx_x == 5 and idx_y == 4:
                assert (
                    z_one_only[idx_x, idx_y]
                    == z_one_only[{"x": idx_x, "y": idx_y}]
                    == 3
                )
            else:
                assert (
                    z_one_only[idx_x, idx_y]
                    == z_one_only[{"x": idx_x, "y": idx_y}]
                    == 0
                )
