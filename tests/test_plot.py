import numpy as np
import pytest

from hist import Hist, NamedHist, axis

plt = pytest.importorskip("matplotlib.pyplot")


def test_general_plot1d():
    """
    Test general plot1d -- whether 1d-Hist can be plotted properly.
    """

    h = Hist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
    ).fill(np.random.normal(size=10))

    assert h.plot1d(color="green", ls="--", lw=3)
    plt.close("all")

    # dimension error
    h = Hist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", label="b [units]", underflow=False, overflow=False
        ),
    ).fill(np.random.normal(size=10), np.random.normal(size=10))

    with pytest.raises(Exception):
        h.plot1d()

    # wrong kwargs names
    with pytest.raises(Exception):
        h.project("A").plot1d(abc="red")

    # wrong kwargs type
    with pytest.raises(Exception):
        h.project("B").plot1d(ls="red")

    plt.close("all")


def test_general_plot2d():
    """
    Test general plot2d -- whether 2d-Hist can be plotted properly.
    """

    h = Hist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", label="b [units]", underflow=False, overflow=False
        ),
    ).fill(np.random.normal(size=10), np.random.normal(size=10))

    assert h.plot2d(cmap="cividis")

    # dimension error
    h = Hist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", label="b [units]", underflow=False, overflow=False
        ),
    ).fill(np.random.normal(size=10), np.random.normal(size=10))

    with pytest.raises(Exception):
        h.project("A").plot2d()

    # wrong kwargs names
    with pytest.raises(Exception):
        h.plot2d(abc="red")

    # wrong kwargs type
    with pytest.raises(Exception):
        h.plot2d(cmap=0.1)

    plt.close("all")


def test_general_plot2d_full():
    """
    Test general plot2d_full -- whether 2d-Hist can be fully plotted properly.
    """

    h = Hist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", label="b [units]", underflow=False, overflow=False
        ),
    ).fill(np.random.normal(size=10), np.random.normal(size=10))

    assert h.plot2d_full(
        main_cmap="cividis",
        top_ls="--",
        top_color="orange",
        top_lw=2,
        side_ls="-.",
        side_lw=1,
        side_color="steelblue",
    )

    assert h.plot2d_full(
        main_cmap="cividis",
        top_kw={
            "ls": "--",
            "color": "orange",
            "lw": 2,
        },
        side_kw={
            "ls": "-.",
            "lw": 1,
            "color": "steelblue",
        },
    )

    # dimension error
    h = Hist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", label="b [units]", underflow=False, overflow=False
        ),
    ).fill(np.random.normal(size=10), np.random.normal(size=10))

    with pytest.raises(Exception):
        h.project("A").plot2d_full()

    # wrong kwargs names
    with pytest.raises(Exception):
        h.plot2d_full(abc="red")

    with pytest.raises(Exception):
        h.plot2d_full(color="red")

    # wrong kwargs type
    with pytest.raises(Exception):
        h.plot2d_full(main_cmap=0.1, side_lw="autumn")

    plt.close("all")


def test_general_plot():
    """
    Test general plot -- whether Hist can be plotted properly.
    """

    h = Hist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
    ).fill(np.random.normal(size=10))

    assert h.plot(color="green", ls="--", lw=3)

    h = Hist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", label="b [units]", underflow=False, overflow=False
        ),
    ).fill(np.random.normal(size=10), np.random.normal(size=10))

    assert h.plot(cmap="cividis")

    # dimension error
    h = Hist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", label="b [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="C", label="c [units]", underflow=False, overflow=False
        ),
    ).fill(
        np.random.normal(size=10), np.random.normal(size=10), np.random.normal(size=10)
    )

    with pytest.raises(Exception):
        h.plot()

    # wrong kwargs names
    with pytest.raises(Exception):
        h.project("A").plot(abc="red")

    with pytest.raises(Exception):
        h.project("A", "C").plot(abc="red")

    # wrong kwargs type
    with pytest.raises(Exception):
        h.project("B").plot(ls="red")

    with pytest.raises(Exception):
        h.project("A", "C").plot(cmap=0.1)

    plt.close("all")


def test_general_plot_pull():
    """
    Test general plot_pull -- whether 1d-Hist can be plotted pull properly.
    """

    np.random.seed(42)

    h = Hist(
        axis.Regular(
            50, -4, 4, name="S", label="s [units]", underflow=False, overflow=False
        )
    ).fill(np.random.normal(size=10))

    def pdf(x, a=1 / np.sqrt(2 * np.pi), x0=0, sigma=1, offset=0):
        return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) + offset

    assert h.plot_pull(
        pdf,
        eb_ecolor="crimson",
        eb_mfc="crimson",
        eb_mec="crimson",
        eb_fmt="o",
        eb_ms=6,
        eb_capsize=1,
        eb_capthick=2,
        eb_alpha=0.8,
        fp_c="chocolate",
        fp_ls="-",
        fp_lw=3,
        fp_alpha=1.0,
        bar_fc="orange",
        pp_num=6,
        pp_fc="orange",
        pp_alpha=0.618,
        pp_ec=None,
    )

    pdf_str = "a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) + offset"

    assert h.plot_pull(pdf_str)

    assert h.plot_pull("gauss")

    assert h.plot_pull("gauss", likelihood=True)

    # dimension error
    hh = Hist(
        axis.Regular(
            50, -4, 4, name="X", label="s [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="Y", label="s [units]", underflow=False, overflow=False
        ),
    ).fill(np.random.normal(size=10), np.random.normal(size=10))

    with pytest.raises(Exception):
        hh.plot_pull(pdf)

    # no eval-able variable
    with pytest.raises(Exception):
        h.plot_pull("1")

    with pytest.raises(Exception):
        h.plot_pull(1)

    with pytest.raises(Exception):
        h.plot_pull(0.1)

    with pytest.raises(Exception):
        h.plot_pull((1, 2))

    with pytest.raises(Exception):
        h.plot_pull([1, 2])

    with pytest.raises(Exception):
        h.plot_pull({"a": 1})

    # wrong kwargs names
    with pytest.raises(Exception):
        h.plot_pull(pdf, abc="crimson", xyz="crimson")

    with pytest.raises(Exception):
        h.plot_pull(pdf, ecolor="crimson", mfc="crimson")

    # not disabled params
    h.plot_pull(pdf, eb_label="value")

    h.plot_pull(pdf, fp_label="value")

    h.plot_pull(pdf, ub_label="value")

    h.plot_pull(pdf, bar_label="value")

    h.plot_pull(pdf, pp_label="value")

    # disabled params
    with pytest.raises(Exception):
        h.plot_pull(pdf, bar_width="value")

    # wrong kwargs types
    with pytest.raises(Exception):
        h.plot_pull(pdf, eb_ecolor=1.0, eb_mfc=1.0)  # kwargs should be str

    plt.close("all")


def test_named_plot1d():
    """
    Test named plot1d -- whether 1d-NamedHist can be plotted properly.
    """

    h = NamedHist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
    ).fill(A=np.random.normal(size=10))

    assert h.plot1d(color="green", ls="--", lw=3)
    plt.close("all")

    # dimension error
    h = NamedHist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", label="b [units]", underflow=False, overflow=False
        ),
    ).fill(B=np.random.normal(size=10), A=np.random.normal(size=10))

    with pytest.raises(Exception):
        h.plot1d()

    # wrong kwargs names
    with pytest.raises(Exception):
        h.project("A").plot1d(abc="red")

    # wrong kwargs type
    with pytest.raises(Exception):
        h.project("B").plot1d(ls="red")

    plt.close("all")


def test_named_plot2d():
    """
    Test named plot2d -- whether 2d-NamedHist can be plotted properly.
    """

    h = NamedHist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", label="b [units]", underflow=False, overflow=False
        ),
    ).fill(B=np.random.normal(size=10), A=np.random.normal(size=10))

    assert h.plot2d(cmap="cividis")
    plt.close("all")

    # dimension error
    h = NamedHist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", label="b [units]", underflow=False, overflow=False
        ),
    ).fill(B=np.random.normal(size=10), A=np.random.normal(size=10))

    with pytest.raises(Exception):
        h.project("A").plot2d()

    # wrong kwargs names
    with pytest.raises(Exception):
        h.plot2d(abc="red")

    # wrong kwargs type
    with pytest.raises(Exception):
        h.plot2d(cmap=0.1)

    plt.close("all")


def test_named_plot2d_full():
    """
    Test named plot2d_full -- whether 2d-NamedHist can be fully plotted properly.
    """

    h = NamedHist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", label="b [units]", underflow=False, overflow=False
        ),
    ).fill(B=np.random.normal(size=10), A=np.random.normal(size=10))

    assert h.plot2d_full(
        main_cmap="cividis",
        top_ls="--",
        top_color="orange",
        top_lw=2,
        side_ls="-.",
        side_lw=1,
        side_color="steelblue",
    )
    plt.close("all")

    # dimension error
    h = NamedHist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", label="b [units]", underflow=False, overflow=False
        ),
    ).fill(B=np.random.normal(size=10), A=np.random.normal(size=10))

    with pytest.raises(Exception):
        h.project("A").plot2d_full()

    # wrong kwargs names
    with pytest.raises(Exception):
        h.plot2d_full(abc="red")

    with pytest.raises(Exception):
        h.plot2d_full(color="red")

    # wrong kwargs type
    with pytest.raises(Exception):
        h.plot2d_full(main_cmap=0.1, side_lw="autumn")

    plt.close("all")


def test_named_plot():
    """
    Test named plot -- whether NamedHist can be plotted properly.
    """

    h = NamedHist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
    ).fill(A=np.random.normal(size=10))

    assert h.plot(color="green", ls="--", lw=3)

    h = NamedHist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", label="b [units]", underflow=False, overflow=False
        ),
    ).fill(B=np.random.normal(size=10), A=np.random.normal(size=10))

    assert h.plot(cmap="cividis")
    plt.close("all")

    # dimension error
    h = NamedHist(
        axis.Regular(
            50, -5, 5, name="A", label="a [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="B", label="b [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="C", label="c [units]", underflow=False, overflow=False
        ),
    ).fill(
        A=np.random.normal(size=10),
        B=np.random.normal(size=10),
        C=np.random.normal(size=10),
    )

    with pytest.raises(Exception):
        h.plot()

    # wrong kwargs names
    with pytest.raises(Exception):
        h.project("A").plot(abc="red")

    with pytest.raises(Exception):
        h.project("A", "C").plot(abc="red")

    # wrong kwargs type
    with pytest.raises(Exception):
        h.project("B").plot(ls="red")

    with pytest.raises(Exception):
        h.project("A", "C").plot(cmap=0.1)

    plt.close("all")


def test_named_plot_pull():
    """
    Test named plot_pull -- whether 1d-NamedHist can be plotted pull properly.
    """

    np.random.seed(42)

    h = NamedHist(
        axis.Regular(
            50, -4, 4, name="S", label="s [units]", underflow=False, overflow=False
        )
    ).fill(S=np.random.normal(size=10))

    def pdf(x, a=1 / np.sqrt(2 * np.pi), x0=0, sigma=1, offset=0):
        return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) + offset

    assert h.plot_pull(
        pdf,
        eb_ecolor="crimson",
        eb_mfc="crimson",
        eb_mec="crimson",
        eb_fmt="o",
        eb_ms=6,
        eb_capsize=1,
        eb_capthick=2,
        eb_alpha=0.8,
        fp_c="chocolate",
        fp_ls="-",
        fp_lw=3,
        fp_alpha=1.0,
        bar_fc="orange",
        pp_num=6,
        pp_fc="orange",
        pp_alpha=0.618,
        pp_ec=None,
    )

    pdf_str = "a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) + offset"

    assert h.plot_pull(pdf_str)

    # dimension error
    hh = NamedHist(
        axis.Regular(
            50, -4, 4, name="X", label="s [units]", underflow=False, overflow=False
        ),
        axis.Regular(
            50, -4, 4, name="Y", label="s [units]", underflow=False, overflow=False
        ),
    ).fill(X=np.random.normal(size=10), Y=np.random.normal(size=10))

    with pytest.raises(Exception):
        hh.plot_pull(pdf)

    # no eval-able variable
    with pytest.raises(Exception):
        h.plot_pull("1")

    with pytest.raises(Exception):
        h.plot_pull(1)

    with pytest.raises(Exception):
        h.plot_pull(0.1)

    with pytest.raises(Exception):
        h.plot_pull((1, 2))

    with pytest.raises(Exception):
        h.plot_pull([1, 2])

    with pytest.raises(Exception):
        h.plot_pull({"a": 1})
    plt.close("all")

    # wrong kwargs names
    with pytest.raises(Exception):
        h.plot_pull(pdf, abc="crimson", xyz="crimson")

    with pytest.raises(Exception):
        h.plot_pull(pdf, ecolor="crimson", mfc="crimson")

    # not disabled params
    h.plot_pull(pdf, eb_label="value")

    h.plot_pull(pdf, fp_label="value")

    h.plot_pull(pdf, ub_label="value")

    h.plot_pull(pdf, bar_label="value")

    h.plot_pull(pdf, pp_label="value")

    # disabled params
    with pytest.raises(Exception):
        h.plot_pull(pdf, bar_width="value")

    # wrong kwargs types
    with pytest.raises(Exception):
        h.plot_pull(pdf, eb_ecolor=1.0, eb_mfc=1.0)  # kwargs should be str

    plt.close("all")


@pytest.mark.parametrize("str_alias", ["normal", "gauss", "gaus"])
@pytest.mark.parametrize("use_likelihood", [True, False])
def test_ratiolike_str_alias(str_alias, use_likelihood):
    """
    Test str alias for callable in plot_ratio and plot_pull
    """

    np.random.seed(42)

    h = NamedHist(
        axis.Regular(
            50, -4, 4, name="S", label="s [units]", underflow=False, overflow=False
        )
    ).fill(S=np.random.normal(size=10))

    assert h.plot_ratio(str_alias, likelihood=use_likelihood)
    assert h.plot_pull(str_alias, likelihood=use_likelihood)


@pytest.mark.mpl_image_compare(baseline_dir="baseline", savefig_kwargs={"dpi": 50})
def test_image_plot_pull():
    """
    Test plot_pull by comparing against a reference image generated via
    `pytest --mpl-generate-path=tests/baseline`
    """

    np.random.seed(42)

    h = Hist(
        axis.Regular(
            50, -4, 4, name="S", label="s [units]", underflow=False, overflow=False
        )
    ).fill(np.random.normal(size=100))

    def pdf(x, a=1 / np.sqrt(2 * np.pi), x0=0, sigma=1, offset=0):
        return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) + offset

    fig = plt.figure()

    assert h.plot_pull(
        pdf,
        eb_color="black",
        fp_color="blue",
        ub_color="lightblue",
        fit_fmt=r"{name} = {value:.3g} $\pm$ {error:.3g}",
    )

    return fig


@pytest.mark.mpl_image_compare(baseline_dir="baseline", savefig_kwargs={"dpi": 50})
def test_image_plot_ratio_hist():
    """
    Test plot_pull by comparing against a reference image generated via
    `pytest --mpl-generate-path=tests/baseline`
    """

    np.random.seed(42)

    hist_1 = Hist(
        axis.Regular(
            50, -5, 5, name="X", label="x [units]", underflow=False, overflow=False
        )
    ).fill(np.random.normal(size=1000))
    hist_2 = Hist(
        axis.Regular(
            50, -5, 5, name="X", label="x [units]", underflow=False, overflow=False
        )
    ).fill(np.random.normal(size=1700))

    fig = plt.figure()

    assert hist_1.plot_ratio(
        hist_2, rp_num_label="numerator", rp_denom_label="denominator"
    )

    return fig


@pytest.mark.mpl_image_compare(baseline_dir="baseline", savefig_kwargs={"dpi": 50})
def test_image_plot_ratio_callable():
    """
    Test plot_pull by comparing against a reference image generated via
    `pytest --mpl-generate-path=tests/baseline`
    """

    np.random.seed(42)

    hist_1 = Hist(
        axis.Regular(
            50, -5, 5, name="X", label="x [units]", underflow=False, overflow=False
        )
    ).fill(np.random.normal(size=1000))

    def model(x, a=1 / np.sqrt(2 * np.pi), x0=0, sigma=1, offset=0):
        return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) + offset

    fig = plt.figure()

    assert hist_1.plot_ratio(
        model, eb_color="black", fp_color="blue", ub_color="lightblue"
    )

    return fig


@pytest.mark.mpl_image_compare(baseline_dir="baseline", savefig_kwargs={"dpi": 50})
def test_plot1d_auto_handling():
    """
    Test plot() by comparing against a reference image generated via
    `pytest --mpl-generate-path=tests/baseline`
    """

    np.random.seed(42)

    h = Hist(
        axis.Regular(10, 0, 10, name="variable", label="variable"),
        axis.StrCategory("", name="dataset", growth=True),
    )

    h_nameless = Hist(
        axis.Regular(10, 0, 10),
        axis.StrCategory("", growth=True),
    )

    h.fill(dataset="A", variable=np.random.normal(3, 2, 100))
    h.fill(dataset="B", variable=np.random.normal(5, 2, 100))
    h.fill(dataset="C", variable=np.random.normal(7, 2, 100))

    h_nameless.fill(np.random.normal(3, 2, 1000), "A")
    h_nameless.fill(np.random.normal(5, 2, 1000), "B")
    h_nameless.fill(np.random.normal(7, 2, 1000), "C")

    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(14, 10))

    assert h.plot(ax=ax1[0])
    assert h_nameless.plot(ax=ax2[0])

    # Discrete axis plotting not yet implemented
    # assert h.plot(ax=ax1[1], overlay='variable')
    # assert h.plot(ax=ax2[1], overlay=1)

    return fig
