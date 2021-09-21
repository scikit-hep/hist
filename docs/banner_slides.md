---
aspectratio: 169
fontfamily: bookman
---

# Simple usage

<!--
Build with:
pandoc docs/banner_slides.md -t beamer -o banner_slides.pdf

Converted to GIF with ezgif.com, 300 ms delay time.
-->

```python
import hist

# Make a histogram
h = hist.Hist(
    hist.axes.Regular(10, 0, 1, name="x")
)

# Fill it with events
h.fill(x=[.2, .3, .6, .9])

# Compute the sum
total = h.sum()
```

# QuickConstruct

```python
from hist import Hist

# Make a histogram
h = Hist.new.Reg(10, 0, 1, name="x").Double()

# Fill it with events
h.fill(x=[.2, .3, .6, .9])

# Compute the sum
total = h.sum()
```

# Plotting: plot_full

\begin{center}
  \includegraphics[width=.6\textwidth]{docs/_images/fullplot_example.png}
\end{center}

# Plotting: plot_ratio

\begin{center}
  \includegraphics[width=.7\textwidth]{docs/_images/ratioplot_example.png}
\end{center}


# Plotting: plot_pie

\begin{center}
  \includegraphics[width=\textwidth]{docs/_images/pieplot_example.png}
\end{center}


# Advanced Indexing

```python
# Slice in data coordinates
sliced_h = h[.5j:1.5j]

# Sum over and rebin easily
smaller_1d = h_2d[sum, 2j]

# Set and access easily
h[...] = np.asarray(prebinned)

# Project and reorder axes
h2 = h.project("x", "y")
```


# Stacks

```python
s = h.stack("cat")
s.plot(stack=True, histtype="fill")
```

\begin{center}
  \includegraphics[width=.7\textwidth]{docs/_images/stackplot_example.png}
\end{center}



# Many Axis types
::: columns
::: {.column width="60%"}

* Regular: evenly or functionally spaced binning
* Variable: arbitrary binning
* Integer: integer binning
* IntCategory: arbitrary integers
* StrCategory: strings
* Boolean: True/False

Most Axis types support optional extras:

* Underflow and/or Overflow bins
* Growth
* Circular (wraparound)

:::
::: {.column width="40%"}

\begin{center}
  \includegraphics[width=1.05\textwidth]{docs/_images/axis_circular.png}
  \includegraphics[width=.7\textwidth]{docs/_images/axis_regular.png}
  \includegraphics[width=.9\textwidth]{docs/_images/axis_variable.png}
  \includegraphics[width=.65\textwidth]{docs/_images/axis_integer.png}
  \includegraphics[width=\textwidth]{docs/_images/axis_category.png}
\end{center}

:::
:::



# Advanced Storages

* Integers
* Floats
* Weighted Sums
* Means
* Weighted Means

Supports the UHI `PlottableHistogram` protocol!

```python
import mplhep
mplhep.histplot(h)
```



# NumPy compatibility

```python
# Drop in faster replacement
H, edges = hist.numpy.histogram(data)

# Drop in threaded faster replacement
H, edges = hist.numpy.histogram(data, threads=4)

# Drop in much faster replacement for 2D+
H, e1, e2 = hist.numpy.histogram2d(xdata, ydata)

# Return a Histogram, convert back to NumPy style
h = hist.numpy.histogram(data, histogram=hist.Hist)
H, edges = h.to_numpy()
```
