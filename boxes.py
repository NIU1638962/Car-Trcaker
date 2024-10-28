# -*- coding: utf-8 -*- noqa
"""
Created on Sun Oct 27 02:44:52 2024

@author: JoelT
"""
import cv2

import numpy as np

from box import Box

from boxes_utils import centroids_distance



class Boxes():

    __slots__ = ("__boxes", "__last_id")

    def __init__(self):
        self.__boxes = {}
        self.__last_id = 0

    def __getitem__(self, key: str):
        """
        Iterate over all Box.

        Parameters
        ----------
        key : string
            ID of a Box.

        Raises
        ------
        KeyError
            Box with given ID not found.

        Returns
        -------
        box : Box
            Box obejct with given the ID.

        """
        try:
            box = self.__boxes[key]
        except KeyError as error:
            raise KeyError(f'"{key}" doesn\'t exist') from error

        return box

    def __iter__(self) -> Box:
        """
        Iterate over all Box objects.

        Yields
        ------
        Box
            Box object.

        """
        for box in self.__boxes.values():
            yield box

    # def add_box(self, xyxy, class_type: str | int | float, confiance: float):
    def add_box(self, xyxy, class_type, confiance: float):
        """
        Add a Box to the data structure.

        Parameters
        ----------
        xyxy : TYPE
            Coordinates xyxy of the box.
        class_type : string or integer or float
            Class type of the box.
        confiance : float
            Confiane of the class detected.

        Returns
        -------
        None.

        """
        new_box = Box(xyxy, class_type, confiance)

        self.__last_id += 1

        identifier = f'ID_{self.__last_id}'

        new_box.identifier = identifier
        self.__boxes[identifier] = new_box
        

    def filter_boxes(self, movement_mask: np.ndarray):
        """
        Filter boxes out that were not displaced in the previous frame.

        Parameters
        ----------
        movement_mask : numpy array
            Mask of pixel that moved in the previous frame.

        Returns
        -------
        None.

        """
        boxes = [box for box in self.__boxes.values()]
        cents= []
        boxs_ids = []
        for box in boxes:
            segment = movement_mask[
                box.xyxy[1]:box.xyxy[3], box.xyxy[0]:box.xyxy[2]
            ]

            area = segment.shape[0]*segment.shape[1]

            movement = cv2.countNonZero(segment)
            

            if area < 0 or movement / area < 0.4:
                del self.__boxes[box.identifier]
            else:
                cents.append(box.centroid)
                boxs_ids.append(box.identifier)
                
        for i in range (len(boxs_ids)):
            for j in range (i+1, len(boxs_ids)):
                dist = centroids_distance(np.array([cents[i]]), np.array([cents[j]]))
                if(dist[0]<4):
                    del self.__boxes[boxs_ids[j]]
            
        
                
    def __len__(self):
        return len(self.__boxes)
