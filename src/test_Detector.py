import argparse
from Detector.Detector import ModelBasedDetector
from detector_utils.get_results import plot_train
parser = argparse.ArgumentParser(
    description="do training"
)

parser.add_argument(
    "--epoch", default=15, dest="epoch", help="number of training epoch"
)

parser.add_argument(
    "--save-prefix",default='ssd_512_test2', dest="save_prefix", help="number of training epoch"
)

parser.add_argument(
    "--start-epoch", default=0, dest="start_epoch", help="number of training epoch"
)

parser.add_argument(
    "--save-interval", default=5, dest="save_interval", help="interval of epoch for saving models"
)

parser.add_argument(
    "--save-plot", default=True, dest="save_plot", help="saving or not the train curve"
)


args = parser.parse_args()

if __name__ == '__main__':
    #train
    #working
    # detector = ModelBasedDetector.from_pretrained(base_model='ssd_512_mobilenet1.0_custom')
    # epochs, ce_loss_list, ce_loss_val, smooth_loss_list, smooth_loss_val, map_list = detector.train(args.save_prefix, args.start_epoch, args.epoch, args.save_interval)
    # plot_train(epochs, ce_loss_list, ce_loss_val, smooth_loss_list, smooth_loss_val, map_list,args.save_prefix, args.save_plot)
    #eval
    # detectorf = ModelBasedDetector.from_finetuned('models/ssd_512_best.params')
    # detectorf.set_tests()
    # detectorf.set_labels_and_scores()
    # detectorf.eval(0.05, True)
    detectorp = ModelBasedDetector.from_finetuned('models/ssd_512_best.params', thresh=0.2)
    score, prediction = detectorp.predict('inputs/EAF3.jpg', 'outputs/')
    print(score, prediction)