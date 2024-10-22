import os
import numpy as np

from typing import List
from ultralytics import YOLO
import cv2
from general_utils import show_image_on_window

class Detector():

    __slots__ = ("__name_model", "__model")

    def __init__(self, path_model: str):
        self.__name_model = os.path.split(path_model)[1]
        self.__model = YOLO(path_model)


    def __get_results(self, image: np.array) -> List:
        return self.__model.predict(source = image, show = False, classes = [2, 5, 7])

    def detect_cars(self, image: np.array) -> np.array:
        """
        Locates the cars and returns bounding boxes.

        Parameters
        ----------
        image : numpy array
            Image represented as a numpy array.

        Returns
        -------
        cropped_image : numpy array
            Image with the cropped license plate represented as a numpy array.

        """
        result = self.__get_results(image)[0]
        img = result.plot()
        
        show_image_on_window(img)
        boxes = result.boxes

        bounding_boxes = np.array(boxes.xyxy, dtype='int32')

        #bounding_box = self.__choose_best_result(bounding_boxes, image)

        return result, bounding_boxes

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

