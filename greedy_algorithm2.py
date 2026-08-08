import argparse

from PIL import Image
import numpy as np
import numba as nb
# import matplotlib.pyplot as plt
from skimage.draw import line, line_aa

from visualize_pattern import get_nail_positions as get_unit_nail_positions


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
    # Transpose to match pygame convention
    image = image.T
    # Normalize to [0, 1]
    image = image.astype('float32')
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



def array_to_image(array):
    """Convert array to a grayscale PIL Image.

    In `array`, 1 indicates max darkness. Invert this to match pygame."""
    array = 255 - (array.clip(0, 1) * 255).astype('uint8')
    return Image.fromarray(array.T)


def get_pixel_nail_positions(num_nails, image_size):
    nails = get_unit_nail_positions(num_nails)
    nails_x = remap_range(nails[:, 0], -1, 1, 0, image_size - 1)
    # Pixel row 0 is the top of the image, but unit y=1 is "up", so the y axis
    # must be flipped relative to x when remapping into pixel space.
    nails_y = remap_range(nails[:, 1], -1, 1, image_size - 1, 0)
    nails = np.column_stack([nails_x, nails_y])
    nails = np.round(nails).astype('int64')
    return nails


def get_line_cache(nails):
    cache = {}
    for i in range(len(nails)):
        for j in range(len(nails)):
            if j != i:
                cache[(i, j)] = line(*nails[i], *nails[j])
    return cache


@nb.njit
def line_loss(reference, target, line_coords_x, line_coords_y):
    overdraw_penalty = 2
    underdraw_penalty = 1
    sum_ = 0
    for i in range(len(line_coords_x)):
        x = line_coords_x[i]
        y = line_coords_y[i]
        reference_pixel = reference[x, y]
        target_pixel = target[x, y]
        diff = reference_pixel - target_pixel
        multiplier = overdraw_penalty if diff < 0 else underdraw_penalty
        sum_ += multiplier * diff

    return sum_


# @profile
def find_best_line(reference, nails, line_cache, start_idx, target, depth, half_circle, prune_factor, banlist=()):
    best_line_score = None
    best_line_score_ignoring_children = None
    best_line_index = None
    limit = len(nails) // 2 if half_circle else len(nails)
    for i in range(1, limit, prune_factor[depth]):
        i_wrapped = (start_idx + i) % len(nails)
        if i_wrapped in banlist:
            # We have visited this nail already in some parent call. We can't
            # add another thread involving this nail because the score might
            # be incorrect.
            continue
        line_coords = line_cache[(start_idx, i_wrapped)]
        score = line_loss(reference, target, *line_coords)
        score_tree = 0
        if depth > 0:
            new_banlist = banlist + (i_wrapped,)
            _, score_tree, _ = find_best_line(reference, nails, line_cache, i_wrapped, target, depth - 1, half_circle, prune_factor, new_banlist)

        if best_line_score is None or score + score_tree > best_line_score:
            best_line_score = score + score_tree
            best_line_score_ignoring_children = score
            best_line_index = i_wrapped
    return best_line_index, best_line_score, best_line_score_ignoring_children


PRUNE_FACTORS = {
    0: {0: 1},
    1: {1: 2, 0: 4},
    2: {2: 2, 1: 4, 0: 8},
}


def find_line_configuration(reference, target, num_nails, depth, half_circle, score_cutoff, ema_alpha):
    image_size = reference.shape[0]
    nails = get_pixel_nail_positions(num_nails, image_size)
    line_cache = get_line_cache(nails)
    current_nail = 0
    recent_score_avg = 1
    prune_factor = PRUNE_FACTORS[depth]

    while True:
        best_line_index, best_line_score, best_line_score_ignoring_children = find_best_line(
            reference=reference,
            nails=nails,
            line_cache=line_cache,
            start_idx=current_nail,
            target=target,
            depth=depth,
            half_circle=half_circle,
            prune_factor=prune_factor,
        )
        rr, cc, val = line_aa(*nails[current_nail], *nails[best_line_index])
        print(f"Drawing line from {current_nail} to {best_line_index}, score {best_line_score_ignoring_children:.2f}")
        recent_score_avg = (1 - ema_alpha) * recent_score_avg + ema_alpha * best_line_score_ignoring_children
        print(f"EMA: {recent_score_avg:.2f}")
        if recent_score_avg < score_cutoff:
            print("Done")
            break
        target[rr, cc] += val * 0.5
        current_nail = best_line_index


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="test_images/chord_test.png")
    parser.add_argument("--nails", type=int, default=400)
    parser.add_argument("--image-size", type=int, default=500)
    parser.add_argument("--depth", type=int, default=2, choices=sorted(PRUNE_FACTORS))
    parser.add_argument("--half-circle", action="store_true", default=True)
    parser.add_argument("--score-cutoff", type=float, default=0)
    parser.add_argument("--ema-alpha", type=float, default=0.5)
    parser.add_argument("--output", default="pygame_output.bmp")
    return parser.parse_args()


def main():
    args = parse_args()
    reference = load_image(args.image, args.image_size)
    target = np.zeros_like(reference)
    find_line_configuration(
        reference, target,
        num_nails=args.nails,
        depth=args.depth,
        half_circle=args.half_circle,
        score_cutoff=args.score_cutoff,
        ema_alpha=args.ema_alpha,
    )
    array_to_image(target).save(args.output)


if __name__ == '__main__':
    main()
