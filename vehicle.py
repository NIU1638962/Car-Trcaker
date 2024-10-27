# -*- coding: utf-8 -*- noqa
"""
Created on Sun Oct 27 14:52:30 2024

@author: Joel Tapia Salvador
"""
import numpy as np

from box import Box


class Vehicle:
    __slots__ = ("__boxes", "__id", "__movement_vector", "__status")

    def __init__(self, identifier: str, starting_box: Box):
        self.__id = identifier
        self.__boxes = [starting_box]
        self.__movement_vector = np.zeros(2)
        self.__status = "Entered"

    def __calculate_movement_vector(self):
        self.__movement_vector = self.__boxes[-1].centroid - \
            self.__boxes[-2].centroid

    def new_box(self, new_box: Box):
        self.__boxes.append(new_box)
        self.__calculate_movement_vector()

    @property
    def last_box(self) -> Box:
        return self.__boxes[-1]
