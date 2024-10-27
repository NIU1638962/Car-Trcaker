# -*- coding: utf-8 -*- noqa
"""
Created on Tue Oct 22 12:30:27 2024

@author: Joan Bernat
"""
import os

import numpy as np

from typing import List
from ultralytics import YOLO

from boxes import Boxes
from general_utils import show_image_on_window


class Detector():

    __slots__ = ("__name_model", "__model")

    def __init__(self, path_model: str, show_results: bool):
        self.__name_model = os.path.split(path_model)[1]

        os.environ['YOLO_VERBOSE'] = str(show_results)

        self.__model = YOLO(path_model, verbose=show_results)

    def __get_results(self, image: np.array) -> List:
        return self.__model.predict(
            source=image,
            show=False,
            classes=(2, 5, 7),
        )

    def detect_cars(
            self,
            image: np.array,
            show_results: bool = False
    ) -> Boxes:
        """
        Locates the cars and returns bounding boxes.

        Parameters
        ----------
        image : numpy array
            Image represented as a numpy array.
        show_results : bool, optional
            Show image with the results of YOLO. The default is False.

        Returns
        -------
        Boxes
            Boxes object with all the bounding boxes dectected.

        """
        result = self.__get_results(image)[0]

        if show_results:

            img = result.plot()

            show_image_on_window(img)

        boxes = Boxes()

        for box in result.boxes:
            boxes.add_box(box.xyxy.squeeze(), box.cls, box.conf)

        return boxes

    @property
    def model(self) -> str:
        """
        Property returning the name of the file the model was loaded from.

        Returns
        -------
        string
            Name of the file where the model was loaded from.

        """
        return self.__name_model


if __name__ == "__main__":
    print(
        '\33[31m' + 'You are executing a module file, execute main instead.'
        + '\33[0m')
