import argparse
import json
import os

import numba as nb
import numpy as np
from PIL import Image
from skimage.draw import line_aa
from tqdm import tqdm

from visualize_pattern import get_nail_positions as get_unit_nail_positions

PRUNE_FACTORS = {
    0: {0: 1},
    1: {1: 2, 0: 4},
    2: {2: 2, 1: 4, 0: 8},
}


def crop_to_circle(array):
    assert array.shape[0] == array.shape[1]
    image_size = array.shape[0]
    image_coords = np.arange(image_size)
    x, y = np.meshgrid(image_coords, image_coords)
    # Enable mask within circle of radius image_size / 2, centered in the middle of the image
    mask = (x - image_size / 2) ** 2 + (y - image_size / 2) ** 2 < (image_size / 2) ** 2
    return np.where(mask, array, 0)


def load_image(filename, image_size):
    """Load image, converting to monochrome."""
    image = Image.open(filename)
    image = image.resize((image_size, image_size))
    image = image.convert("L")
    image = np.asarray(image)
    # Transpose to match (x, y) pixel indexing used below
    image = image.T
    # Normalize to [0, 1]
    image = image.astype("float32")
    image /= 255
    # Make 1 maximum darkness instead of maximum brightness
    image = 1 - image
    # Crop to circle
    image = crop_to_circle(image)
    return image


def remap_range(array, input_low, input_high, output_low, output_high):
    input_range = input_high - input_low
    output_range = output_high - output_low
    mult = output_range / input_range
    return (array - input_low) * mult + output_low


def get_pixel_nail_positions(num_nails, image_size):
    """Nail positions in pixel coordinates, matching the physical nail layout
    used elsewhere (n=0 is up).

    Pixel row 0 is the top of the image, but unit y=1 is "up", so the y axis
    must be flipped relative to x when remapping into pixel space.
    """
    nails = get_unit_nail_positions(num_nails)
    nails_x = remap_range(nails[:, 0], -1, 1, 0, image_size - 1)
    nails_y = remap_range(nails[:, 1], -1, 1, image_size - 1, 0)
    nails = np.column_stack([nails_x, nails_y])
    return np.round(nails).astype("int64")


@nb.njit(cache=True)
def bresenham_line(x0, y0, x1, y1):
    """Yield the pixel coordinates from (x0, y0) to (x1, y1) inclusive.

    Matches skimage.draw.line's pixel walk exactly."""
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        yield x, y
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


@nb.njit(cache=True)
def line_loss(reference, target, x0, y0, x1, y1):
    """Sum of overdraw/underdraw-penalized pixel diffs along the line from
    (x0, y0) to (x1, y1)."""
    overdraw_penalty = 2
    underdraw_penalty = 1
    sum_ = 0.0
    for x, y in bresenham_line(x0, y0, x1, y1):
        diff = reference[x, y] - target[x, y]
        multiplier = overdraw_penalty if diff < 0 else underdraw_penalty
        sum_ += multiplier * diff
    return sum_


@nb.njit(cache=True)
def find_best_line(reference, nails_x, nails_y, start_idx, target, depth, half_circle, prune_factor, banlist, banlist_len):
    num_nails = len(nails_x)
    best_line_score = 0.0
    best_line_score_ignoring_children = 0.0
    best_line_index = -1
    have_best = False
    limit = num_nails // 2 if half_circle else num_nails
    for i in range(1, limit, prune_factor[depth]):
        i_wrapped = (start_idx + i) % num_nails
        banned = False
        for k in range(banlist_len):
            if banlist[k] == i_wrapped:
                # We have visited this nail already in some parent call. We can't
                # add another thread involving this nail because the score might
                # be incorrect.
                banned = True
                break
        if banned:
            continue
        score = line_loss(reference, target, nails_x[start_idx], nails_y[start_idx], nails_x[i_wrapped], nails_y[i_wrapped])
        score_tree = 0.0
        if depth > 0:
            banlist[banlist_len] = i_wrapped
            _, score_tree, _ = find_best_line(
                reference=reference,
                nails_x=nails_x,
                nails_y=nails_y,
                start_idx=i_wrapped,
                target=target,
                depth=depth - 1,
                half_circle=half_circle,
                prune_factor=prune_factor,
                banlist=banlist,
                banlist_len=banlist_len + 1,
            )

        if not have_best or score + score_tree > best_line_score:
            have_best = True
            best_line_score = score + score_tree
            best_line_score_ignoring_children = score
            best_line_index = i_wrapped
    return best_line_index, best_line_score, best_line_score_ignoring_children


def find_line_configuration(reference, num_nails, depth, half_circle, score_cutoff, ema_alpha):
    image_size = reference.shape[0]
    nails = get_pixel_nail_positions(num_nails, image_size)
    nails_x = np.ascontiguousarray(nails[:, 0])
    nails_y = np.ascontiguousarray(nails[:, 1])
    target = np.zeros_like(reference)
    prune_factor = np.zeros(depth + 1, dtype="int64")
    for level, step in PRUNE_FACTORS[depth].items():
        prune_factor[level] = step
    banlist = np.zeros(depth, dtype="int64")

    current_nail = 0
    path = [current_nail]
    recent_score_avg = 1

    with tqdm(desc="Placing threads", unit=" thread") as pbar:
        while True:
            best_line_index, _, best_line_score_ignoring_children = find_best_line(
                reference=reference,
                nails_x=nails_x,
                nails_y=nails_y,
                start_idx=current_nail,
                target=target,
                depth=depth,
                half_circle=half_circle,
                prune_factor=prune_factor,
                banlist=banlist,
                banlist_len=0,
            )
            rr, cc, val = line_aa(*nails[current_nail], *nails[best_line_index])
            recent_score_avg = (1 - ema_alpha) * recent_score_avg + ema_alpha * best_line_score_ignoring_children
            pbar.set_postfix(ema_loss=f"{recent_score_avg:.2f}", cutoff=score_cutoff)
            pbar.update(1)
            if recent_score_avg < score_cutoff:
                break
            target[rr, cc] += val * 0.5
            current_nail = best_line_index
            path.append(current_nail)

    return path


def save_pattern(output_path, num_nails, image_path, image_size, path):
    pattern = {
        "nails": num_nails,
        "image": image_path,
        "image_size": image_size,
        "path": path,
    }
    with open(output_path, "w") as f:
        json.dump(pattern, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--nails", type=int, default=400)
    parser.add_argument("--image-size", type=int, default=500)
    parser.add_argument("--depth", type=int, default=2, choices=sorted(PRUNE_FACTORS))
    parser.add_argument("--half-circle", action="store_true", default=True)
    parser.add_argument("--score-cutoff", type=float, default=0)
    parser.add_argument("--ema-alpha", type=float, default=0.5)
    parser.add_argument("--output")
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output or os.path.join("patterns", os.path.splitext(os.path.basename(args.image))[0] + ".json")

    reference = load_image(args.image, args.image_size)
    path = find_line_configuration(
        reference=reference,
        num_nails=args.nails,
        depth=args.depth,
        half_circle=args.half_circle,
        score_cutoff=args.score_cutoff,
        ema_alpha=args.ema_alpha,
    )
    save_pattern(output, args.nails, args.image, args.image_size, path)
    print(f"Wrote {len(path)} nails to {output}")


if __name__ == "__main__":
    main()
