import fire
import time 
from Detector.Detector import ModelBasedDetector
from detector_utils.get_results import plot_train


class Predictor(object):

    def train_from_pretrained(self, batch_size=10, data_path='../Data/EAF_real', save_prefix='ssd_512',
                              start_epoch=0, epochs=10, save_plot=True, model_folder='models/', train_result_folder='results_train/',
                              log_folder='logs/'):
        '''
        command line to train from the pretrained ssd_512_mobile_net model
        
        Args:
        - batch_size (int): the batch size for training
        - data_path (str): the root of the VOC folder containing train set
        - save_prefix (str): the prefix name used for saving output file
        - start_epoch (int): the epoch from which we start the training
        - epoch (int): the number of training epochs
        - save_plot (bool): flag to notice saving training plot 
        - train_result_folder (str): the folder in which we save train plot
        - log_folder (str): the folder in which we save logs
        '''
        start = time.time()
        detector = ModelBasedDetector.from_pretrained(data_path=data_path, batch_size=batch_size, base_model='ssd_512_mobilenet1.0_custom', save_prefix=save_prefix)
        epochs, ce_loss_list, ce_loss_val, smooth_loss_list, smooth_loss_val, map_list = detector.train(start_epoch, epochs, log_folder)
        plot_train(epochs, ce_loss_list, ce_loss_val, smooth_loss_list, smooth_loss_val, map_list, save_prefix, train_result_folder, save_plot)
        end = time.time()
        print("the training took " + str(end - start) + " seconds")

    def train_from_finetuned(self, batch_size=10, data_path='../Data/EAF_real', save_prefix='ssd_512',
                             model_name='models/ssd_512_best.params', start_epoch=0, epochs=10, save_plot=True,
                             train_result_folder='results_train/', log_folder='logs/'):
        '''
        command line to train from a finetuned model
        - batch_size (int): the batch size for training
        - batch_size (int): the batch size for training
        - data_path (str): the root of the VOC folder containing train set
        - save_prefix (str): the prefix name used for saving output file
        - model_name (str): the path of the model
        - start_epoch (int): the epoch from which we start the training
        - epoch (int): the number of training epochs
        - save_plot (bool): flag to notice saving training plot
        - train_result_folder (str): the folder in which we save train plot
        - log_folder (str): the folder in which we save logs
        '''
        start = time.time()
        detector = ModelBasedDetector.from_finetuned(model_name, data_path=data_path, save_prefix=save_prefix, batch_size=batch_size)
        epochs, ce_loss_list, ce_loss_val, smooth_loss_list, smooth_loss_val, map_list = detector.train(start_epoch, epochs, log_folder)
        plot_train(epochs, ce_loss_list, ce_loss_val, smooth_loss_list, smooth_loss_val, map_list, save_prefix, train_results_folder, save_plot)
        end = time.time()
        print("the training took " + str(end - start) + " seconds")

    def eval(self, data_path_test='../Data/EAF_real', save_prefix='ssd_512_test2',
             model_path='models/ssd_512_best.params', taux_fp=0.05, save_plot=True,
             results_folder='results_ROC/', log_foler='logs/'):
        '''
        command line to evaluate the model on a test dataset
        Args:
        - data_path_test (str): the root of the VOC folder containing the test set
        - save_prefix (str): the prefix name used for saving output files
        - model_name (str): the path of the model
        - taux_fp (float): the false positive rate we want to obtain the optimal threshold
        - save_plot (bool): flag to notice saving ROC curve plot 
        - results_folder (str): the folder in which we save ROC curve
        - log_folder (str): the folder in which we save logs
        '''
        detector = ModelBasedDetector.from_finetuned(model_path, data_path_test, save_prefix=save_prefix)
        detector._set_tests()
        detector._set_labels_and_scores()
        detector.eval(taux_fp, save_plot)

    def predict(self, model_name='models/ssd_512_best.params', input_path='inputs/EAF3.jpg', output_folder='outputs/', thresh=0.2, data_path_test='../Data/EAF_real'):
        '''
        command line to predict with an input image and save the result 
        
        Args:
        - model_name (str): the path of the model
        - input_path (str): the path of the input image
        - output_path (str): the folder in which the output is saved
        - thresh (float): the threshold to determine if there is a protection regarde the model score
        '''
        detectorp = ModelBasedDetector.from_finetuned(model_name, thresh=thresh)
        score, prediction, box_coord = detectorp.predict(input_path, output_folder)
        print(score, prediction, box_coord)



if __name__ == '__main__':
    fire.Fire(Predictor)