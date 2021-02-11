import unittest
import os
from pathlib import Path
from src.Detector.Detector import ModelBasedDetector
from src.detector_utils.get_results import plot_train
from gluoncv import model_zoo
import numpy as np
class CliDetectorTestCase(unittest.TestCase):
  
  def setUp(self):
    Path('tests/test_data/detector_data').mkdir(parents=True, exist_ok=True)
    Path('tests/test_data/detector_data/models').mkdir(parents=True, exist_ok=True)
    if os.path.exists('tests/test_data/detector_data/logs'):
      for f in os.listdir('tests/test_data/detector_data/logs'):
        os.remove(os.path.join('tests/test_data/detector_data/logs', f))
      os.rmdir('tests/test_data/detector_data/logs')
    if os.path.exists('tests/test_data/detector_data/results_train'):
      for f in os.listdir('tests/test_data/detector_data/results_train'):
        os.remove(os.path.join('tests/test_data/detector_data/results_train', f))
      os.rmdir('tests/test_data/detector_data/results_train')
    if os.path.exists('tests/test_data/detector_data/results'):
      for f in os.listdir('tests/test_data/detector_data/results'):
        os.remove(os.path.join('tests/test_data/detector_data/results', f))
      os.rmdir('tests/test_data/detector_data/results')
    if os.path.exists('tests/test_data/detector_data/models'):
      for f in os.listdir('tests/test_data/detector_data/models'):
        if f=='test_best.params':
          os.remove(os.path.join('tests/test_data/detector_data/models', f))
    if os.path.exists('tests/test_data/detector_data/outputs'):
      for f in os.listdir('tests/test_data/detector_data/outputs'):
        os.remove(os.path.join('tests/test_data/detector_data/outputs', f))
      os.rmdir('tests/test_data/detector_data/outputs')
    self.detector1 = ModelBasedDetector.from_pretrained(data_path_test='tests/test_data/detector_data/for_test', data_path='tests/test_data/detector_data/for_test',
                                                       batch_size=10, base_model='ssd_512_mobilenet1.0_custom', save_prefix='test')
    name_model = os.listdir('models/')
    if not name_model:
      raise NameError('Veuillez déposer un modèle dans le dossier detector_data/models')
    name = [n for n in name_model if n.endswith(".params")]
    self.detector2 = ModelBasedDetector.from_finetuned('models/' + name[0], data_path_test='tests/test_data/detector_data/for_test', save_prefix='test')

  def test_train_from_pretrained(self):
    epochs, ce_loss_list, ce_loss_val, smooth_loss_list, smooth_loss_val, map_list = self.detector1.train(0,1, log_folder='tests/test_data/detector_data/logs/',
                                                                                                         model_folder='tests/test_data/detector_data/models/')
    plot_train(epochs, ce_loss_list, ce_loss_val, smooth_loss_list, smooth_loss_val, map_list, 'test', 'tests/test_data/detector_data/results_train/')
    self.assertNotEqual(ce_loss_list, [])
    self.assertNotEqual(ce_loss_val, [])
    self.assertNotEqual(smooth_loss_list, [])
    self.assertNotEqual(smooth_loss_val, [])
    self.assertNotEqual(map_list, [])
    name_train_result = os.listdir('tests/test_data/detector_data/results_train/')
    name_log = os.listdir('tests/test_data/detector_data/logs/')
    self.assertIn('test_best_map.log', name_log)
    self.assertIn('test_train.log', name_log)
    self.assertEqual(name_train_result[0],'test_train_curves.png')

  def test_eval(self):
    name_model = os.listdir('models/')
    name = [n for n in name_model if n.endswith(".params")]
    detector2 = ModelBasedDetector.from_finetuned('models/' + name[0],data_path_test='tests/test_data/detector_data/for_test', save_prefix='test')
    self.detector2._set_tests()
    self.detector2._set_labels_and_scores('tests/test_data/detector_data/logs/')
    self.detector2.eval(0.5, True, 'tests/test_data/detector_data/results/', 'tests/test_data/detector_data/logs/')
    name_log = os.listdir('tests/test_data/detector_data/logs/')
    name_results = os.listdir('tests/test_data/detector_data/results/')
    print(detector2.y_true, detector2.y_scores)
    self.assertIn('test_ROC_curve.png', name_results)
    self.assertIn('eval.log', name_log)
    self.assertIn('eval.json', name_log)

  def test_predict(self):
    score, prediction, box_coord = self.detector2.predict('tests/test_data/detector_data/images_test/EAF17.jpg',
                                                          'tests/test_data/detector_data/outputs/')
    output_name = os.listdir('tests/test_data/detector_data/outputs/')
    self.assertIn('EAF17.jpg', output_name)
    self.assertNotEqual(box_coord, [])


  def test_from_pretrained_net_type(self):
      net = model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes=['cheminee', 'eaf'], pretrained_base=False, transfer='voc')
      self.assertEqual(type(self.detector1.net), type(net))