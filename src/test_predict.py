import argparse
from detector_utils.get_results import load_model, get_ouput_model, get_prediction, process_output_img
parser = argparse.ArgumentParser(
    description="evaluate a model"
)

parser.add_argument(
    "--model-name", default='ssd_512_best', dest="model_name", help="the path of images"
)

parser.add_argument(
    "--name", required=True, dest="name", help="the name of image"
)
args = parser.parse_args()

if __name__ == '__main__':
    thresh = 0.2925
    score, output_path = get_ouput_model('inputs/'+args.name +'.jpg', "models/"+args.model_name+".params", thresh, 'test', True)
    #process_output_img(output_path)
    predict = get_prediction(score, thresh)
    print(predict)