# -*- coding: utf-8 -*-
import sys
import shutil
import argparse
import hist
import boost_histogram as bh
from histoprint import print_hist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, help="input file to read from (stdin by default)"
    )
    parser.add_argument("-b", "--buckets", type=int, help="number of bins", default=20)
    parser.add_argument(
        "-s",
        "--screen-width",
        type=int,
        help="maximum screen width",
        default=shutil.get_terminal_size()[0],
    )
    parser.add_argument("-t", "--label", type=str, help="label for plot")
    parser.add_argument("-o", "--output-image", type=str, help="save image to file")
    args = parser.parse_args()

    print(
        "Classic hist interface - please use histoprint instead; this supports multiple file formats and much more!"
    )

    with open(args.input) if args.input else sys.stdin as f:
        values = [float(v) for v in f]

    h = bh.numpy.histogram(values, bins=args.buckets, histogram=hist.Hist)

    if args.output_image:
        import matplotlib.pyplot as plt

        h.plot()
        plt.savefig(args.output_image)
    else:
        print_hist(
            h,
            label=args.label,
            summary=True,
        )


if __name__ == "__main__":
    main()
