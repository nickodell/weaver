import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from table_config import TABLE_RADIUS_MM
from visualize_pattern import get_nail_positions

STEP_DELAY = 1
THREAD_ALPHA = 0.8
THREAD_WIDTH = 0.5


def step_length_m(nails, current_pin, next_pin):
    diff = nails[next_pin] - nails[current_pin]
    return np.linalg.norm(diff) * TABLE_RADIUS_MM / 1000


def send_pin(pin):
    """Placeholder for sending a move command over serial."""
    print(f"Would move to pin {pin}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern")
    return parser.parse_args()


def load_pattern(path):
    with open(path) as f:
        pattern = json.load(f)
    return get_nail_positions(pattern["nails"]), pattern["path"]


def setup_plot(nails):
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(nails[:, 0], nails[:, 1], color="lightgray", s=2, zorder=2)
    ax.set_aspect("equal")
    ax.axis("off")

    drawn_line, = ax.plot([], [], color="black", alpha=THREAD_ALPHA, linewidth=THREAD_WIDTH)
    next_line, = ax.plot([], [], color="red", alpha=THREAD_ALPHA, linewidth=THREAD_WIDTH)

    plt.show(block=False)
    return fig, drawn_line, next_line


def redraw(fig):
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def run_steps(nails, path, fig, drawn_line, next_line):
    drawn_x = [nails[path[0], 0]]
    drawn_y = [nails[path[0], 1]]

    steps = list(zip(path, path[1:]))
    total_m = sum(step_length_m(nails, a, b) for a, b in steps)

    with tqdm(total=total_m, unit="m", unit_scale=False, bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f}m [{elapsed}<{remaining}]") as pbar:
        for current_pin, next_pin in steps:
            next_line.set_data(
                [nails[current_pin, 0], nails[next_pin, 0]],
                [nails[current_pin, 1], nails[next_pin, 1]],
            )
            redraw(fig)
            send_pin(next_pin)
            time.sleep(STEP_DELAY)

            drawn_x.append(nails[next_pin, 0])
            drawn_y.append(nails[next_pin, 1])
            drawn_line.set_data(drawn_x, drawn_y)
            next_line.set_data([], [])
            redraw(fig)

            step_m = step_length_m(nails, current_pin, next_pin)
            pbar.set_postfix_str(f"+{step_m:.2f}m")
            pbar.update(step_m)


def main():
    args = parse_args()
    nails, path = load_pattern(args.pattern)
    fig, drawn_line, next_line = setup_plot(nails)
    run_steps(nails, path, fig, drawn_line, next_line)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
