# -*- coding: utf-8 -*- noqa
"""
Created on Sun Oct 27 16:21:40 2024

@author: Joel Tapia Salvador
"""
import numpy as np

from queue import PriorityQueue

from boxes import Boxes
from boxes_utils import centroids_distance
from vehicle import Vehicle


class Vehicles():
    __slots__ = ("__previous_mapping", "__vehicles")

    def __init__(self):
        self.__vehicles = {}

    def process(self, current_boxes, previous_boxes):
        ordered_distances = {}

        for current_box in current_boxes:
            ordered_distances[current_box.identifier] = PriorityQueue()

            for previous_box in previous_boxes:
                dist = centroids_distance(
                    np.array([current_box.centroid]), np.array([previous_box.centroid]))[0, 0]

                ordered_distances[current_box.identifier].put_nowait(
                    (dist, previous_box.identifier))

        # TODO
