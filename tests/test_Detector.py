import unittest

from gluoncv import model_zoo
from src.Detector.Detector import ModelBasedDetector


class ModelBasedDetectorTestCase(unittest.TestCase):
    def setUp(self):
        self.detector1 = ModelBasedDetector.from_pretrained(data_path='../Data/EAF_real', batch_size=10, base_model='ssd_512_mobilenet1.0_custom', save_prefix='ssd_512')
        self.detector2 = ModelBasedDetector.from_finetuned('../models/ssd_512_best.params',data_path='../Data/EAF_real', batch_size=10, base_model='ssd_512_mobilenet1.0_custom', save_prefix='ssd_512')

    def test_from_pretrained_att(self):
        net = model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes=['cheminee', 'eaf'], pretrained_base=False, transfer='voc')
        self.assertEqual(type(self.detector1.net), type(net))

if __name__ == '__main__':
    unittest.main()