import numpy as np

from hist import Hist, Stack, axis

# ToDo: specify what error is raised


def test_stack_init():
    """
    Test stack init -- whether Stack can be properly initialized.
    """

    # basic
    h1 = Hist(axis.Regular(10, 0, 1, name="x")).fill(x=np.random.normal(size=10))

    h2 = Hist(axis.Regular(10, 0, 1, name="x")).fill(x=np.random.normal(size=10))

    st = Stack(h1, h2)
    st.plot()
