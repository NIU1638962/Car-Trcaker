# -*- coding: utf-8 -*- noqa
"""
Created on Sun Oct 27 02:18:38 2024

@author: Joel Tapia Salvador
"""
from typing import Tuple


class Box():

    __slots__ = ("__centroid", "__class_type", "__confiance", "__id", "__xyxy")

    def __init__(self, xyxy, class_type: str | int | float, confiance: float):
        self.__xyxy = tuple(
            [
                int(coordinate) for coordinate, control in zip(xyxy, range(4))
            ]
        )

        if isinstance(class_type, str):
            self.__class_type = class_type
        else:
            mapping = {0: 'person',
                       1: 'bicycle',
                       2: 'car',
                       3: 'motorcycle',
                       4: 'airplane',
                       5: 'bus',
                       6: 'train',
                       7: 'truck',
                       8: 'boat',
                       9: 'traffic light',
                       10: 'fire hydrant',
                       11: 'stop sign',
                       12: 'parking meter',
                       13: 'bench',
                       14: 'bird',
                       15: 'cat',
                       16: 'dog',
                       17: 'horse',
                       18: 'sheep',
                       19: 'cow',
                       20: 'elephant',
                       21: 'bear',
                       22: 'zebra',
                       23: 'giraffe',
                       24: 'backpack',
                       25: 'umbrella',
                       26: 'handbag',
                       27: 'tie',
                       28: 'suitcase',
                       29: 'frisbee',
                       30: 'skis',
                       31: 'snowboard',
                       32: 'sports ball',
                       33: 'kite',
                       34: 'baseball bat',
                       35: 'baseball glove',
                       36: 'skateboard',
                       37: 'surfboard',
                       38: 'tennis racket',
                       39: 'bottle',
                       40: 'wine glass',
                       41: 'cup',
                       42: 'fork',
                       43: 'knife',
                       44: 'spoon',
                       45: 'bowl',
                       46: 'banana',
                       47: 'apple',
                       48: 'sandwich',
                       49: 'orange',
                       50: 'broccoli',
                       51: 'carrot',
                       52: 'hot dog',
                       53: 'pizza',
                       54: 'donut',
                       55: 'cake',
                       56: 'chair',
                       57: 'couch',
                       58: 'potted plant',
                       59: 'bed',
                       60: 'dining table',
                       61: 'toilet',
                       62: 'tv',
                       63: 'laptop',
                       64: 'mouse',
                       65: 'remote',
                       66: 'keyboard',
                       67: 'cell phone',
                       68: 'microwave',
                       69: 'oven',
                       70: 'toaster',
                       71: 'sink',
                       72: 'refrigerator',
                       73: 'book',
                       74: 'clock',
                       75: 'vase',
                       76: 'scissors',
                       77: 'teddy bear',
                       78: 'hair drier',
                       79: 'toothbrush'}

            self.__class_type = mapping[int(class_type)]

        self.__confiance = float(confiance)

        self.__calculate_centroid()

    def __calculate_centroid(self):
        self.__centroid = tuple(
            [
                (self.__xyxy[2] + self.__xyxy[0]) / 2,
                (self.__xyxy[3] + self.__xyxy[1]) / 2
            ]
        )

    @property
    def centroid(self) -> Tuple[int, int]:
        """
        Get centroid of the box.

        Returns
        -------
        Tuple[integer, integer]
            Centroid of the box.

        """
        return self.__centroid

    @property
    def class_type(self) -> str:
        """
        Get class type of the box.

        Returns
        -------
        string
            Class type of the box.

        """
        return self.__class_type

    @property
    def confiance(self) -> float:
        """
        Get confiance of the box.

        Returns
        -------
        float
            Confiance of the box.

        """
        return self.__confiance

    @property
    def identifier(self) -> str:
        """
        Get identifier of the box.

        Returns
        -------
        string
            Identifier of the box.

        """
        return self.__id

    @identifier.setter
    def identifier(self, value: str):
        """
        Set identifier of the box.

        Parameters
        ----------
        value : string
            New value of the identifier.

        Returns
        -------
        None.

        """
        self.__id = value

    @property
    def xyxy(self) -> Tuple[int, int, int, int]:
        """
        Get xyxyx coordinates of the box.

        Returns
        -------
        Tuple[int, int, int, int]
            xyxy coordinated of the box.

        """
        return self.__xyxy
