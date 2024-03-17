from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt
import pygame
import threading
import time
from skimage.draw import line, line_aa

running = True


def load_image(filename):
    """Load image, converting to monochrome."""
    image = Image.open(filename)
    image = image.resize((500, 500))
    image = image.convert("L")
    image = np.asarray(image)
    # Normalize to [0, 1]
    image = image.astype('float32')
    image /= 255
    # Make 1 maximum darkness instead of maximum brightness
    image = 1 - image
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


def get_nail_positions():
    image_pixels = 500
    num_nails = 300
    nails = np.linspace(0, 2*np.pi, num_nails)
    nails_x = np.cos(nails)
    nails_y = np.sin(nails)
    nails_x = remap_range(nails_x, -1, 1, 0, image_pixels - 1)
    nails_y = remap_range(nails_y, -1, 1, 0, image_pixels - 1)
    nails = np.column_stack([nails_x, nails_y])
    nails = np.round(nails).astype('int64')
    return nails


def find_best_line(reference, nails, start_idx, target, half_circle=False):
    best_line_score = None
    best_line_index = None
    for i in range(1,len(nails)):
        i_wrapped = (start_idx + i) % len(nails)
        line_coords = line(*nails[start_idx], *nails[i_wrapped])
        score = (reference[line_coords] - target[line_coords]).mean()
        if best_line_score is None or score > best_line_score:
            best_line_score = score
            best_line_index = i_wrapped
    return best_line_index, best_line_score



def find_line_configuration(reference, target):
    global running
    nails = get_nail_positions()
    current_nail = 0
    
    while running:
        # target[:] = 0
        # time.sleep(1)
        best_line_index, best_line_score = find_best_line(reference, nails, current_nail, target)
        rr, cc, val = line_aa(*nails[current_nail], *nails[best_line_index])
        print(f"Drawing line from {current_nail} to {best_line_index}, score {best_line_score:.2f}")
        if best_line_score < 0:
            print("Done")
            break
        target[rr, cc] += val * 0.5
        current_nail = best_line_index


        # time.sleep(0.1)
            # time.sleep(0.1)
            # breakpoint()
            # out
            # pass

        # out[:] = 1
        # time.sleep(1)


def main():
    global running
    pygame.init()
    display = pygame.display.set_mode((1000, 500))
    reference = load_image("test_images/circle_pattern.png")
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

