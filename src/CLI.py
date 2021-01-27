import fire
import time 
from Detector.Detector import ModelBasedDetector
from detector_utils.get_results import plot_train
class Predictor(object):

    def train_from_pretrained(self, batch_size=40,data_path='../Data/EAF_real', save_prefix='ssd_512', start_epoch=0, epoch=10, save_plot=True):
        start = time.time()
        detector = ModelBasedDetector.from_pretrained(data_path=data_path, base_model='ssd_512_mobilenet1.0_custom', save_prefix=save_prefix,batch_size=batch_size)
        epochs, ce_loss_list, ce_loss_val, smooth_loss_list, smooth_loss_val, map_list = detector.train(start_epoch, epoch)
        plot_train(epochs, ce_loss_list, ce_loss_val, smooth_loss_list, smooth_loss_val, map_list, save_prefix, save_plot)
        end = time.time()
        print("the training took " + str(end - start) + " seconds")

    def train_from_finetuned(self, batch_size=10, data_path='../Data/EAF_real', save_prefix='ssd_512', model_name='models/ssd_512_best.params', start_epoch=0, epoch=10, save_plot=True):
        start = time.time()
        detector = ModelBasedDetector.from_finetuned(model_name, data_path, save_prefix=save_prefix, batch_size=batch_size)
        epochs, ce_loss_list, ce_loss_val, smooth_loss_list, smooth_loss_val, map_list = detector.train(start_epoch, epoch)
        plot_train(epochs, ce_loss_list, ce_loss_val, smooth_loss_list, smooth_loss_val, map_list, save_prefix, save_plot)
        end = time.time()
        print("the training took " + str(end - start) + " seconds")

    def eval(self, data_path_test='../Data/EAF_real', save_prefix='ssd_512_test2', model_name='models/ssd_512_best.params', taux_fp=0.05, save_plot=True):
        detector = ModelBasedDetector.from_finetuned(model_name, data_path_test, save_prefix=save_prefix)
        detector.set_tests()
        detector.set_labels_and_scores()
        detector.eval(taux_fp, save_plot)

    def predict(self, model_name='models/ssd_512_best.params', input_path='inputs/EAF3.jpg', output_folder='outputs/'):
        detectorp = ModelBasedDetector.from_finetuned(model_name, thresh=0.2)
        score, prediction = detectorp.predict(input_path, output_folder)
        print(score, prediction)

    def synthetic_train(self):
        pass


if __name__ == '__main__':
    fire.Fire(Predictor)