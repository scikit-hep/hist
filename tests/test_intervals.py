import numpy as np
import pytest
from pytest import approx

from hist import Hist, axis, intervals


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


def test_ratio_uncertainty(hist_fixture):
    hist_1, hist_2 = hist_fixture

    uncertainty_min, uncertainty_max = intervals.ratio_uncertainty(
        hist_1.values(), hist_2.values(), uncertainty_type="poisson"
    )

    assert approx(uncertainty_min) == np.array(
        [
            0.1439794096271186,
            0.12988019998066708,
            0.0711565635066328,
            0.045722288708959336,
            0.04049103990124614,
            0.038474711321686006,
            0.045227104349518155,
            0.06135954973309016,
            0.12378460125991042,
            0.19774186117590858,
        ]
    )

    assert approx(uncertainty_max) == np.array(
        [
            0.22549817680979262,
            0.1615766277480729,
            0.07946632561746425,
            0.04954668134626106,
            0.04327624938437291,
            0.04106267733757407,
            0.04891233040201837,
            0.06909296140898324,
            0.1485919630151803,
            0.2817958228477908,
        ]
    )

    with pytest.raises(TypeError):
        intervals.ratio_uncertainty(
            hist_1.values(), hist_2.values(), uncertainty_type="fail"
        )
