import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import serial_control
from table_config import TABLE_RADIUS_MM
from visualize_pattern import get_nail_positions

STEP_DELAY = 1
THREAD_ALPHA = 0.8
THREAD_WIDTH = 0.5


def step_length_m(nails, current_pin, next_pin):
    diff = nails[next_pin] - nails[current_pin]
    return np.linalg.norm(diff) * TABLE_RADIUS_MM / 1000


def send_pin(ser, pin):
    print(f"Moving to pin {pin}")
    if ser is None:
        time.sleep(STEP_DELAY)
    else:
        serial_control.send_pin(ser, pin)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern")
    parser.add_argument("--dry-run", action="store_true", help="print moves instead of sending them over serial")
    return parser.parse_args()


def load_pattern(path):
    with open(path) as f:
        pattern = json.load(f)
    return get_nail_positions(pattern["nails"]), pattern["path"]


def setup_plot(nails, pattern_name):
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(nails[:, 0], nails[:, 1], color="lightgray", s=2, zorder=2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{pattern_name} - 0%")

    drawn_line, = ax.plot([], [], color="black", alpha=THREAD_ALPHA, linewidth=THREAD_WIDTH)
    next_line, = ax.plot([], [], color="red", alpha=THREAD_ALPHA, linewidth=THREAD_WIDTH)

    plt.show(block=False)
    return fig, ax, drawn_line, next_line


def redraw(fig):
    fig.canvas.draw()
    fig.canvas.flush_events()


def run_steps(nails, path, fig, ax, drawn_line, next_line, ser, pattern_name):
    drawn_x = [nails[path[0], 0]]
    drawn_y = [nails[path[0], 1]]

    steps = list(zip(path, path[1:]))
    total_m = sum(step_length_m(nails, a, b) for a, b in steps)
    done_m = 0

    with tqdm(total=total_m, unit="m", unit_scale=False, bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f}m [{elapsed}<{remaining}]") as pbar:
        for current_pin, next_pin in steps:
            next_line.set_data(
                [nails[current_pin, 0], nails[next_pin, 0]],
                [nails[current_pin, 1], nails[next_pin, 1]],
            )
            redraw(fig)
            send_pin(ser, next_pin)

            drawn_x.append(nails[next_pin, 0])
            drawn_y.append(nails[next_pin, 1])
            drawn_line.set_data(drawn_x, drawn_y)
            next_line.set_data([], [])

            step_m = step_length_m(nails, current_pin, next_pin)
            done_m += step_m
            percent_done = 100 * done_m / total_m
            ax.set_title(f"{pattern_name} - {percent_done:.0f}%")
            redraw(fig)

            pbar.set_postfix_str(f"+{step_m:.2f}m")
            pbar.update(step_m)


def main():
    args = parse_args()
    pattern_name = os.path.basename(args.pattern)
    nails, path = load_pattern(args.pattern)
    fig, ax, drawn_line, next_line = setup_plot(nails, pattern_name)
    ser = None if args.dry_run else serial_control.init_serial()
    run_steps(nails, path, fig, ax, drawn_line, next_line, ser, pattern_name)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
