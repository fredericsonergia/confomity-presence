import argparse
import os
import logging
from utils import get_results
import gluoncv as gcv
from Trainer.trainer_ssd import load_data_VOC

parser = argparse.ArgumentParser(
    description="evaluate a model"
)

parser.add_argument(
    "--model-name", default='ssd_512_best.params', dest="model_name", help="the name of the model"
)

parser.add_argument(
    "--save-plot", default=True, dest="save_plot", help="flag to indicate to save plot"
)

parser.add_argument(
    "--taux-fp", default=0.05, dest="taux_fp", help="rate of false positive we want"
)

parser.add_argument(
    "--save-prefix",default='logs/ssd_512', dest="save_prefix", help="number of training epoch"
)

args = parser.parse_args()


if __name__ == '__main__':
    save_plot = args.save_plot
    _, val_dataset = load_data_VOC()
    test_img_list = get_results.get_test_set()
    model_name = 'models/' + args.model_name
    net = get_results.load_model(model_name)
    y_true, y_scores, mean_iou = get_results.get_labels_and_scores(val_dataset, test_img_list, net)
    opti_thresh = get_results.ROC_curve_thresh(y_true, y_scores, args.taux_fp, save_plot)
    print(f"L'inteserction over union moyen est de {mean_iou}")
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_results.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    logger.info("{len(y_true)}")
    logger.info(f"L'inteserction over union moyen est de {mean_iou}")
    logger.info(f"Le seuil optimal est {opti_thresh} pour un taux de faux position accepté de {args.taux_fp}")