import argparse
import keras_ocr
import json
from conformity.Conformity import Conformity
pipeline = pipeline = keras_ocr.pipeline.Pipeline()

my_parser = argparse.ArgumentParser(description="Json giving information about a conformity file.")
my_parser.add_argument('--save', help="save the printed result into a json file", action="store_true")
my_parser.add_argument('image_path', type= str, help="Path of image to check conformity for.")

args = my_parser.parse_args()

image_path = args.image_path

my_conformity = Conformity(pipeline, image_path)

result = my_conformity.get_conformity()
print(result)

if args.save:
    json_file_name = image_path.replace('.','_')
    json_file_name = json_file_name+'.json'
    with open(json_file_name, 'w') as json_file:
        json.dump(result, json_file)