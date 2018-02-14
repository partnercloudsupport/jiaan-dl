# # -*- coding: utf-8 -*-
"""
Inference event
"""
import json

class InferEvent(object):
    """Class: InferEvent
    """
    def __init__(self, img, model='resnet', network='ssd'):
        """An event definition during inference

        Args:
            img (cv2.Image): Frame capture
            model (str): Name of the model
            network (str): Network used to train model
        """
        self._img = img
        self._model = model
        self._network = network
        self._vectors = []

    @property
    def yscale(self):
        """Difference of shape and training parameters
        """
        return float(self._img.shape[0]/300)

    @property
    def xscale(self):
        """Difference of shape and training parameters
        """
        return float(self._img.shape[1]/300)

    @property
    def vectors(self):
        """Collection of vectors
        """
        return self._vectors

    def add_vector(self, vector, label):
        """Add a new vector

        Args:
            vector (tuple): Coordinates of inferred object
            label (tuple): Name and probability of inferred object
        """
        self._vectors.append({
            'label': label[0],
            'probability': label[1],
            'bbox': (vector.xmin, vector.xmax, vector.ymin, vector.ymax,)
        })

    def __str__(self):
        return json.dumps({
            'frame': {
                'model': self._model,
                'network': self._network,
                'shape': (self.xscale * 300, self.yscale * 300)
            },
            'vectors': self._vectors
        })