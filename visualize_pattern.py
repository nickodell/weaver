import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def get_nail_positions(num_nails):
    angles = np.linspace(0, 2 * np.pi, num_nails, endpoint=False)
    # n=0 is up
    return np.column_stack([-np.sin(angles), np.cos(angles)])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern")
    parser.add_argument("--output")
    parser.add_argument("--thread-alpha", type=float, default=0.8)
    parser.add_argument("--thread-width", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.pattern) as f:
        pattern = json.load(f)

    nails = get_nail_positions(pattern["nails"])
    path = pattern["path"]

    fig, ax = plt.subplots(figsize=(6, 6))

    thread_points = nails[path]
    ax.plot(
        thread_points[:, 0],
        thread_points[:, 1],
        color="black",
        alpha=args.thread_alpha,
        linewidth=args.thread_width,
    )

    ax.scatter(nails[:, 0], nails[:, 1], color="lightgray", s=2, zorder=2)

    ax.set_aspect("equal")
    ax.axis("off")

    if args.output:
        fig.savefig(args.output, dpi=200, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
