import argparse
import sys

from style_transfer.style_transfer import transfer_random_style_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--style_folder",
        help="path to existing folder where images are to be used to be style references",
    )
    parser.add_argument(
        "-c",
        "--content_folder",
        help="path to existing folder where images are to be used to be content references",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        help="path to existing folder where styled images will be put",
    )
    args = parser.parse_args()

    STYLE_FOLDER = args.style_folder
    CONTENT_FOLDER = args.content_folder
    OUTPUT_FOLDER = args.output_folder

    output_image_size = 512
    transfer_random_style_folder(
        CONTENT_FOLDER, STYLE_FOLDER, OUTPUT_FOLDER, output_image_size
    )

