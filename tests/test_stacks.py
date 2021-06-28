from hist import Hist, Stack, axis

# ToDo: specify what error is raised


def test_stack_init():
    """
    Test stack init -- whether Stack can be properly initialized.
    """

    # basic
    h1 = Hist(axis.Regular(10, 0, 1, name="x"), axis.Regular(10, 0, 1, name="y")).fill(
        x=[0.35, 0.35, 0.45], y=[0.35, 0.35, 0.45]
    )

    h2 = Hist(axis.Regular(10, 0, 1, name="x"), axis.Regular(10, 0, 1, name="y")).fill(
        x=[0.25, 0.25, 0.35], y=[0.25, 0.25, 0.35]
    )

    st = Stack(h1, h2)
    st.plot()
