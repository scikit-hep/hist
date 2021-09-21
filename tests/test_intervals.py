from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

from hist import Hist, axis

pytest.importorskip("scipy")
intervals = pytest.importorskip("hist.intervals")


@pytest.fixture(scope="session")
def hist_fixture():
    np.random.seed(42)

    hist_1 = Hist(
        axis.Regular(
            10, -3, 3, name="X", label="x [units]", underflow=False, overflow=False
        )
    ).fill(np.random.normal(size=1000))
    hist_2 = Hist(
        axis.Regular(
            10, -3, 3, name="X", label="x [units]", underflow=False, overflow=False
        )
    ).fill(np.random.normal(size=1700))

    return hist_1, hist_2


def test_poisson_interval(hist_fixture):
    hist_1, hist_2 = hist_fixture

    interval_min, interval_max = intervals.poisson_interval(
        hist_1.values(), hist_2.values()
    )

    assert approx(interval_min) == np.array(
        [
            1.5300291697782136,
            15.163319280895097,
            70.76628228209115,
            138.53885398885032,
            206.76205061547802,
            216.04895966121967,
            146.23699499970348,
            59.34874385941129,
            23.42140089960769,
            4.398298999080774,
        ]
    )
    assert approx(interval_max) == np.array(
        [
            12.599700199673102,
            28.738493673101413,
            94.88918823365604,
            173.30954997605485,
            246.94963052163627,
            257.713403993323,
            181.58237748187338,
            84.74029590412256,
            38.20780361508876,
            13.75903516119368,
        ]
    )

    interval_min, interval_max = intervals.poisson_interval(np.arange(4))
    assert approx(interval_min) == np.array([0.0, 0.17275378, 0.70818544, 1.36729531])
    assert approx(interval_max) == np.array(
        [1.84102165, 3.29952656, 4.63785962, 5.91818583]
    )


def test_clopper_pearson_interval(hist_fixture):
    hist_1, _ = hist_fixture

    interval_min, interval_max = intervals.clopper_pearson_interval(
        hist_1.values() * 0.8, hist_1.values()
    )

    assert approx(interval_min) == np.array(
        [
            0.4757493739959414,
            0.6739341864914268,
            0.745848569184471,
            0.7626112365367469,
            0.769793269861182,
            0.7705165046817444,
            0.7636693080808218,
            0.7409965720106119,
            0.6996942106437167,
            0.5617068220945686,
        ]
    )

    assert approx(interval_max) == np.array(
        [
            0.9660393063397832,
            0.8909499770371633,
            0.8458913477820733,
            0.8331466664953922,
            0.8273390876155183,
            0.8267415410793807,
            0.832305086850061,
            0.8493900886167576,
            0.8763181169809199,
            0.9404385982116935,
        ]
    )


def test_ratio_uncertainty():
    num, denom = np.meshgrid(np.array([0, 1, 4, 512]), np.array([0, 1, 4, 512]))

    uncertainty_min, uncertainty_max = intervals.ratio_uncertainty(
        num, denom, uncertainty_type="poisson"
    )

    assert approx(uncertainty_min, nan_ok=True) == np.array(
        [
            [np.nan, np.nan, np.nan, np.nan],
            [0.0, 8.27246221e-01, 1.91433919e00, 2.26200365e01],
            [0.0, 2.06811555e-01, 4.78584797e-01, 5.65500911e00],
            [0.0, 1.61571528e-03, 3.73894372e-03, 4.41797587e-02],
        ]
    )

    assert approx(uncertainty_max, nan_ok=True) == np.array(
        [
            [np.nan, np.nan, np.nan, np.nan],
            [np.nan, 2.29952656e00, 3.16275317e00, 2.36421589e01],
            [np.nan, 5.74881640e-01, 7.90688293e-01, 5.91053972e00],
            [np.nan, 4.49126281e-03, 6.17725229e-03, 4.61760915e-02],
        ]
    )

    uncertainty_min, uncertainty_max = intervals.ratio_uncertainty(
        num, denom, uncertainty_type="poisson-ratio"
    )

    assert approx(uncertainty_min, nan_ok=True) == np.array(
        [
            [np.nan, np.inf, np.inf, np.inf],
            [0.0, 9.09782858e-01, 3.09251539e00, 3.57174304e02],
            [0.0, 2.14845433e-01, 6.11992834e-01, 5.67393184e01],
            [0.0, 1.61631629e-03, 3.75049626e-03, 6.24104251e-02],
        ]
    )

    assert approx(uncertainty_max, nan_ok=True) == np.array(
        [
            [np.nan, np.nan, np.nan, np.nan],
            [5.30297438e00, 1.00843679e01, 2.44458061e01, 2.45704433e03],
            [5.84478627e-01, 8.51947064e-01, 1.57727199e00, 1.18183919e02],
            [3.60221785e-03, 4.50575120e-03, 6.22048393e-03, 6.65647601e-02],
        ]
    )

    with pytest.raises(ValueError):
        intervals.ratio_uncertainty(num, denom, uncertainty_type="efficiency")

    uncertainty_min, uncertainty_max = intervals.ratio_uncertainty(
        np.minimum(num, denom), denom, uncertainty_type="efficiency"
    )

    assert approx(uncertainty_min, nan_ok=True) == np.array(
        [
            [np.nan, np.nan, np.nan, np.nan],
            [0.0, 0.8413447460685429, 0.8413447460685429, 0.8413447460685429],
            [0.0, 0.207730893696323, 0.36887757085042716, 0.36887757085042716],
            [0.0, 0.0016157721916044239, 0.003735294987003171, 0.0035892884494188593],
        ]
    )
    assert approx(uncertainty_max, nan_ok=True) == np.array(
        [
            [np.nan, np.nan, np.nan, np.nan],
            [0.8413447460685429, 0.0, 0.0, 0.0],
            [0.3688775708504272, 0.36840242550395996, 0.0, 0.0],
            [0.0035892884494188337, 0.004476807721636625, 0.006134065381665161, 0.0],
        ]
    )

    with pytest.raises(TypeError):
        intervals.ratio_uncertainty(num, denom, uncertainty_type="fail")


def test_valid_efficiency_ratio_uncertainty(hist_fixture):
    """
    Test that the upper bound for the error interval does not exceed unity
    for efficiency ratio plots.
    """

    hist_1, _ = hist_fixture
    num = hist_1.values()
    den = num

    efficiency_ratio = num / den
    _, uncertainty_max = intervals.ratio_uncertainty(
        num, den, uncertainty_type="efficiency"
    )
    efficiency_err_up = efficiency_ratio + uncertainty_max

    assert len(efficiency_err_up[efficiency_err_up > 1.0]) == 0
