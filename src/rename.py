import os 
import argparse
from detector_utils.VOC_form import rename_img, add_text


parser = argparse.ArgumentParser(
    description="rename pictures"
)

parser.add_argument(
    "--path", required=True, dest="path", help="the path of images"
)

parser.add_argument(
    "--name", default='', dest="name", help="the name of images"
)

parser.add_argument(
    "--start", default='', dest="start", help="the number from which the annotation begin"
)

args = parser.parse_args()

if args.name:
    rename_img(args.path, args.name, args.start)

