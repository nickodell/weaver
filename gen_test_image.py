import argparse

from PIL import Image, ImageDraw

from greedy_algorithm import get_pixel_nail_positions


def draw_chords(image_size, num_nails, pins, line_width):
    image = Image.new("L", (image_size, image_size), color=255)
    draw = ImageDraw.Draw(image)
    nails = get_pixel_nail_positions(num_nails, image_size)

    points = [tuple(nails[pin]) for pin in pins]
    draw.line(points, fill=0, width=line_width)
    return image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=500)
    parser.add_argument("--nails", type=int, default=400)
    parser.add_argument("--pins", type=int, nargs="+", default=[0, 133, 267, 0])
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--output", default="test_images/chord_test.png")
    return parser.parse_args()


def main():
    args = parse_args()
    image = draw_chords(args.image_size, args.nails, args.pins, args.line_width)
    image.save(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
