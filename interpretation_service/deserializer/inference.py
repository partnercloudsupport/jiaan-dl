# # -*- coding: utf-8 -*-
"""
Inference Event emitted from IoT query
"""
class InferenceEvent(object):
    def __init__(self, event=None):
        self._event = event or {}

    @property
    def vectors(self):
        return self._event.get('vectors', [])

    @property
    def metadata(self):
        return self._event.get('frame', {})