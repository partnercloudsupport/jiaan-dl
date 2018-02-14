# # -*- coding: utf-8 -*-
"""
Vector
"""
import json

class Vector(object):
    """Class: Vector
    """
    def __init__(self, points, xscale=1, yscale=1):
        self._points = points
        self._xscale = xscale
        self._yscale = yscale

    @property
    def xmin(self):
        """x-min
        """
        return int(self._xscale * self._points[0]) + int(self._points[0])

    @property
    def ymin(self):
        """y-min
        """
        return int(self._yscale * self._points[2])

    @property
    def xmax(self):
        """x-max
        """
        return int(self._xscale * self._points[1]) + int(self._points[1])

    @property
    def ymax(self):
        """y-max
        """
        return int(self._yscale * self._points[3])

    def __str__(self):
        return json.dumps((self.xmin, self.xmax, self.ymin, self.ymax,))