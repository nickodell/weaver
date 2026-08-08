import argparse
import json
import os
import time

import numpy as np
from tqdm import tqdm

import serial_control
from live_preview import LivePreview
from table_config import TABLE_RADIUS_MM
from visualize_pattern import get_nail_positions

STEP_DELAY = 1


def step_length_m(nails, current_pin, next_pin):
    diff = nails[next_pin] - nails[current_pin]
    return np.linalg.norm(diff) * TABLE_RADIUS_MM / 1000


def send_pin(ser, pin):
    print(f"Moving to pin {pin}")
    if ser is None:
        time.sleep(STEP_DELAY)
    else:
        serial_control.send_pin(ser, pin)


def send_home(ser):
    print("Homing")
    if ser is None:
        time.sleep(STEP_DELAY)
    else:
        serial_control.send_home(ser)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern")
    parser.add_argument("--dry-run", action="store_true", help="print moves instead of sending them over serial")
    parser.add_argument("--debug-serial", action="store_true", help="print raw bytes read from and written to the serial port")
    return parser.parse_args()


def load_pattern(path):
    with open(path) as f:
        pattern = json.load(f)
    return get_nail_positions(pattern["nails"]), pattern["path"]


def run_steps(nails, path, live_preview, ser, pattern_name):
    steps = list(zip(path, path[1:]))
    total_m = sum(step_length_m(nails, a, b) for a, b in steps)
    done_m = 0

    with tqdm(total=total_m, unit="m", unit_scale=False, bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f}m [{elapsed}<{remaining}]") as pbar:
        for current_pin, next_pin in steps:
            live_preview.show_next(nails[current_pin, 0], nails[current_pin, 1], nails[next_pin, 0], nails[next_pin, 1])
            send_pin(ser, next_pin)

            live_preview.add_point(nails[next_pin, 0], nails[next_pin, 1], refresh=False)
            live_preview.clear_next()

            step_m = step_length_m(nails, current_pin, next_pin)
            done_m += step_m
            percent_done = 100 * done_m / total_m
            live_preview.set_title(f"{pattern_name} - {percent_done:.0f}%")
            live_preview.refresh()

            pbar.set_postfix_str(f"+{step_m:.2f}m")
            pbar.update(step_m)


def main():
    args = parse_args()
    serial_control.debug = args.debug_serial
    pattern_name = os.path.basename(args.pattern)
    nails, path = load_pattern(args.pattern)
    live_preview = LivePreview(nails, f"{pattern_name} - 0%")
    ser = None if args.dry_run else serial_control.init_serial()
    send_home(ser)
    run_steps(nails, path, live_preview, ser, pattern_name)
    live_preview.finish()


if __name__ == "__main__":
    main()
