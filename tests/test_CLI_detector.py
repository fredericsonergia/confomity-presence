import unittest
import os
from pathlib import Path
from src.Detector.Detector import ModelBasedDetector
from src.detector_utils.get_results import plot_train
from gluoncv import model_zoo

class CliDetectorTestCase(unittest.TestCase):
  
  def setUp(self):
    Path('test_data/detector_data').mkdir(parents=True, exist_ok=True)
    Path('test_data/detector_data/models').mkdir(parents=True, exist_ok=True)
    if os.path.exists('tests/test_data/detector_data/logs'):
      for f in os.listdir('tests/test_data/detector_data/logs'):
        os.remove(os.path.join('tests/test_data/detector_data/logs', f))
      os.rmdir('tests/test_data/detector_data/logs')
    if os.path.exists('tests/test_data/detector_data/results_train'):
      for f in os.listdir('tests/test_data/detector_data/results_train'):
        os.remove(os.path.join('tests/test_data/detector_data/results_train', f))
      os.rmdir('tests/test_data/detector_data/results_train')
    if os.path.exists('tests/test_data/detector_data/models'):
      for f in os.listdir('tests/test_data/detector_data/models'):
        if f=='test_best.params':
          os.remove(os.path.join('tests/test_data/detector_data/models', f))
    self.detector = ModelBasedDetector.from_pretrained(data_path_test='tests/test_data/detector_data/EAF_test', data_path='tests/test_data/detector_data/EAF_test',
                                                       batch_size=10, base_model='ssd_512_mobilenet1.0_custom', save_prefix='test')

  def test_from_pretrained_net_type(self):
      net = model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes=['cheminee', 'eaf'], pretrained_base=False, transfer='voc')
      self.assertEqual(type(self.detector.net), type(net))

  def test_train_from_pretrained(self):
    name_logs = ['test_train.log', 'test_best_map.log']
    epochs, ce_loss_list, ce_loss_val, smooth_loss_list, smooth_loss_val, map_list = self.detector.train(0,1, log_folder='tests/test_data/detector_data/logs/',
                                                                                                         model_folder='tests/test_data/detector_data/models/')
    plot_train(epochs, ce_loss_list, ce_loss_val, smooth_loss_list, smooth_loss_val, map_list, 'test', 'tests/test_data/detector_data/results_train/')
    self.assertNotEqual(ce_loss_list, [])
    self.assertNotEqual(ce_loss_val, [])
    self.assertNotEqual(smooth_loss_list, [])
    self.assertNotEqual(smooth_loss_val, [])
    self.assertNotEqual(map_list, [])
    name_train_result = os.listdir('tests/test_data/detector_data/results_train/')
    name_log = os.listdir('tests/test_data/detector_data/logs/')
    self.assertIn(name_log[0],name_logs)
    self.assertIn(name_log[1],name_logs)
    self.assertEqual(name_train_result[0],'test_train_curves.png')

  def test_eval(self):
    self.detector._set_tests()
    self.detector._set_labels_and_scores()
    self.detector.eval(0.2, True)
    name_log = os.listdir('tests/test_data/detector_data/logs/')
    self.assertIn('test_ROC_curve.png', name_log)