import fire
from Detector.Detector import ModelBasedDetector
from utils.get_results import plot_train
class Predictor(object):

    def train(self, save_prefix= 'ssd_512_test2', start_epoch=0, epoch=15, save_interval=5, save_plot=True):
        detector = ModelBasedDetector.from_pretrained(base_model='ssd_512_mobilenet1.0_custom', save_prefix=save_prefix)
        epochs, ce_loss_list, ce_loss_val, smooth_loss_list, smooth_loss_val, map_list = detector.train(start_epoch, epoch, save_interval)
        plot_train(epochs, ce_loss_list, ce_loss_val, smooth_loss_list, smooth_loss_val, map_list, save_prefix, save_plot)

    def eval(self,save_prefix='ssd_512_test2', model_name='models/ssd_512_best.params', taux_fp=0.05, save_plot=True):
        detector = ModelBasedDetector.from_finetuned(model_name, save_prefix=save_prefix)
        detector.set_tests()
        detector.set_labels_and_scores()
        detector.eval(taux_fp, save_plot)

    def predict(self, model_name='models/ssd_512_best.params', input_path='inputs/EAF3.jpg', output_folder='outputs/'):
        detectorp = ModelBasedDetector.from_finetuned(model_name, thresh=0.2)
        score, prediction = detectorp.predict(input_path, output_folder)
        print(score, prediction)


if __name__ == '__main__':
    fire.Fire(Predictor)