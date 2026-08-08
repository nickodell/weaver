import argparse
import json

NAILS = 400
IMAGE = "<generated>"
IMAGE_SIZE = 500


def build_path(num_triangles):
    path = []
    for offset in range(num_triangles):
        points = [
            offset,
            offset + round(NAILS / 3),
            offset + round(2 * NAILS / 3),
        ]
        path.extend(p % NAILS for p in points)
    return path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-triangles", type=int, default=10)
    parser.add_argument("--output", default="patterns/test_pattern.json")
    return parser.parse_args()


def main():
    args = parse_args()
    pattern = {
        "nails": NAILS,
        "image": IMAGE,
        "image_size": IMAGE_SIZE,
        "path": build_path(args.num_triangles),
    }
    with open(args.output, "w") as f:
        json.dump(pattern, f, indent=2)


if __name__ == "__main__":
    main()
