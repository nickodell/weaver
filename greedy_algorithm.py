from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt
import pygame
import threading
import time
from skimage.draw import line, line_aa

running = True


def crop_to_circle(array):
    assert array.shape[0] == array.shape[1]
    image_size = array.shape[0]
    image_coords = np.arange(500)
    x, y = np.meshgrid(image_coords, image_coords)
    # Enable mask within circle of radius image_size / 2, centered in the middle of the image
    mask = (x - image_size / 2) ** 2 + (y - image_size / 2) ** 2 < (image_size / 2) ** 2
    return np.where(mask, array, 0)


def load_image(filename):
    """Load image, converting to monochrome."""
    image = Image.open(filename)
    image_size = 500
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
    # print(np.asarray(image).shape)
    # print(np.asarray(image))


def remap_range(array, input_low, input_high, output_low, output_high):
    input_range = input_high - input_low
    output_range = output_high - output_low
    mult = output_range / input_range
    return (array - input_low) * mult + output_low



def create_monochrome_surf(shape):
    surf = pygame.Surface(shape, depth=8)
    surf.set_palette([(i, i, i) for i in range(256)])
    return surf


def copy_to_surface(surf, array):
    """Copy array to pygame.Surface

    In `array`, 1 indicates max darkness.
    Invert this to match pygame."""
    array = 255 - (array.clip(0, 1) * 255).astype('uint8')
    array_access = pygame.surfarray.pixels2d(surf)
    array_access[:] = array
    # Remove A to unlock surface
    del array_access


def get_nail_positions(image_pixels, num_nails):
    nails = np.linspace(0, 2*np.pi, num_nails)
    nails_x = np.cos(nails)
    nails_y = np.sin(nails)
    nails_x = remap_range(nails_x, -1, 1, 0, image_pixels - 1)
    nails_y = remap_range(nails_y, -1, 1, 0, image_pixels - 1)
    nails = np.column_stack([nails_x, nails_y])
    nails = np.round(nails).astype('int64')
    return nails


def line_loss(reference_pixels, target_pixels):
    # return (reference_pixels - target_pixels).mean()
    # return (reference_pixels - target_pixels).sum()
    diff = reference_pixels - target_pixels
    overdraw = np.where(diff < 0, diff, 0)
    underdraw = np.where(diff > 0, diff, 0)
    overdraw_penalty = 2
    underdraw_penalty = 1

    return (overdraw_penalty * overdraw.sum() + underdraw_penalty * underdraw.sum()) # / len(reference_pixels)


def find_best_line(reference, nails, start_idx, target, depth, half_circle, prune_factor, banlist=()):
    best_line_score = None
    best_line_score_ignoring_children = None
    best_line_index = None
    limit = len(nails) // 2 if half_circle else len(nails)
    if prune_factor is None:
        # Disable pruning
        prune_factor = {0: 1}
    for i in range(1, limit, prune_factor[depth]):
        i_wrapped = (start_idx + i) % len(nails)
        if i_wrapped in banlist:
            # We have visited this nail already in some parent call. We can't
            # add another thread involving this nail because the score might
            # be incorrect.
            continue
        line_coords = line(*nails[start_idx], *nails[i_wrapped])
        score = line_loss(reference[line_coords], target[line_coords])
        score_tree = 0
        if depth > 0:
            new_banlist = banlist + (i_wrapped,)
            _, score_tree, _ = find_best_line(reference, nails, i_wrapped, target, depth - 1, half_circle, prune_factor, new_banlist)

        if best_line_score is None or score + score_tree > best_line_score:
            best_line_score = score + score_tree
            best_line_score_ignoring_children = score
            best_line_index = i_wrapped
    return best_line_index, best_line_score, best_line_score_ignoring_children



def find_line_configuration(reference, target):
    global running
    num_nails = 300
    image_pixels = reference.shape[0]
    nails = get_nail_positions(image_pixels, num_nails)
    current_nail = 0
    recent_score_avg = 1
    ema_alpha = 0.5
    score_cutoff = 0
    
    while running:
        # target[:] = 0
        # time.sleep(1)
        depth = 2
        if depth == 0:
            prune_factor = {0: 1}
        elif depth == 1:
            prune_factor = {1: 2, 0: 4}
        elif depth == 2:
            prune_factor = {2: 2, 1: 4, 0: 8}
        else:
            raise Exception()
        half_circle = True
        best_line_index, best_line_score, best_line_score_ignoring_children = find_best_line(
            reference=reference,
            nails=nails,
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


def main():
    global running
    pygame.init()
    display = pygame.display.set_mode((1000, 500))
    reference = load_image("test_images/portrait.jpg")
    target = np.zeros_like(reference)
    reference_surf = create_monochrome_surf(reference.shape)
    target_surf = create_monochrome_surf(target.shape)
    clock = pygame.time.Clock()
    line_conf_thread = threading.Thread(target=find_line_configuration, args=(reference, target))
    line_conf_thread.start()
    copy_to_surface(reference_surf, reference)
    display.blit(reference_surf, (0, 0))
    while running:
        copy_to_surface(target_surf, target)
        display.blit(target_surf, (500, 0))
        pygame.display.update()

        # Sleep till next frame
        clock.tick(60)

        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    pygame.quit()


if __name__ == '__main__':
    # load_image("test_images/circle_pattern.png")
    main()

