# # -*- coding: utf-8 -*-
import unittest
from mock import Mock

from model import InferEvent, Vector

class VectorTestSuite(unittest.TestCase):
    def setUp(self):
        self._vector = Vector((1, 2, 1, 2), 1, 1)

    def test_xmin(self):
        self.assertEqual(self._vector.xmin, 2, 'Invalid X-min')
        
    def test_xmax(self):
        self.assertEqual(self._vector.xmax, 4, 'Invalid X-max')

    def test_ymin(self):
        self.assertEqual(self._vector.ymin, 1, 'Invalid Y-min')

    def test_ymax(self):
        self.assertEqual(self._vector.ymax, 2, 'Invalid Y-max')

class InferEventTestSuite(unittest.TestCase):
    def setUp(self):
        img = Mock()
        img.shape = [300, 300]
        self._event = InferEvent(img)

    def test_yscale(self):
        self.assertEqual(self._event.yscale, 1, 'Invalid Y-scale')

    def test_xscale(self):
        self.assertEqual(self._event.xscale, 1, 'Invalid X-scale')

    def test_vectors(self):
        v1 = Vector((1, 2, 1, 2), 1, 1)
        self._event.add_vector(v1, ('test', 0.55,))

        self.assertEqual(self._event.vectors, [{'bbox': (2, 4, 1, 2), 'probability': 0.55, 'label': 'test'}], 'Vectors are invalid')

if __name__ == '__main__':
    unittest.main()