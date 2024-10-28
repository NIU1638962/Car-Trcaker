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
import uuid


class Vehicles():
    __slots__ = ("__last_id", "__previous_boxes",
                 "__previous_mapping", "__vehicles", "__up", "__down")

    def __init__(self):
        self.__vehicles = {}
        self.__previous_mapping = {}
        self.__previous_boxes = Boxes()
        self.__last_id = 0
        self.__up = 0
        self.__down = 0

    def __add_vehicle(self, box):
        self.__last_id += 1
        identifier = f'Vehicle_{self.__last_id}'
        self.__vehicles[identifier] = Vehicle(identifier, box)

        return identifier
    
    def process(self, new_boxes, controller):
        for new_box in new_boxes:
            min_id = None
            min_dist = 50
            for box in self.__previous_boxes:
                dist = centroids_distance(np.array([new_box.centroid]), np.array([box.centroid]))
                if (dist[0] < min_dist):
                    min_dist = dist
                    min_id = box.identifier
                    new_box.assigned = True
            if(new_box.assigned):
                self.__vehicles[new_box.identifier] = self.__previous_mapping[min_id]
            else:
                self.__vehicles[new_box.identifier] = uuid.uuid4()
                if(new_box.centroid[1]>260):
                    if(controller.control):
                        if(abs(controller.centroid[0]-new_box.centroid[0])>30):
                            self.__up+=1
                            controller.add_centroid(new_box.centroid)
                    else:
                        self.__up+=1
                        controller.add_centroid(new_box.centroid)
                    
        if(len(new_boxes)<len(self.__previous_boxes)):
            last_uuid = set(self.__previous_mapping.values())
            new_uuid = set(self.__vehicles.values())
            x_id = last_uuid - new_uuid
            x_id = list(x_id)[0]
            for box_id, v_id in self.__previous_mapping.items():
                if(v_id==x_id):
                    if(self.__previous_boxes[box_id].centroid[1]>270):
                        if(controller.control):
                            if(abs(controller.centroid[0]-self.__previous_boxes[box_id].centroid[0])>30):
                                self.__down+=1
                                controller.add_centroid(self.__previous_boxes[box_id].centroid)
                        else:
                            self.__down+=1
                            controller.add_centroid(self.__previous_boxes[box_id].centroid)
                            
                            
        self.__previous_mapping = self.__vehicles
        self.__previous_boxes = new_boxes
        self.__vehicles = {}
        
    @property
    def up(self):
        return self.__up
        
    @property
    def down(self):
        return self.__down                
                    
    @property
    def previous_boxes(self):
        return self.__previous_boxes
    
    @property
    def previous_mapping(self):
        return self.__previous_mapping
            
          
        
    """
    def process(self, new_boxes):
        ordered_distances = {}

        for new_box in new_boxes:
            ordered_distances[new_box.identifier] = PriorityQueue()

            for previous_box in self.__previous_boxes:
                dist = centroids_distance(
                    np.array([new_box.centroid]), np.array([previous_box.centroid]))[0, 0]

                ordered_distances[new_box.identifier].put_nowait(
                    (dist, previous_box.identifier))

        if len(self.__previous_mapping) == 0:
            for box_identifier in ordered_distances.keys():
                self.__previous_mapping[box_identifier] = self.__add_vehicle(
                    new_boxes[box_identifier])
        else:
            new_mapping = {}

            temp_inverse_mapping = {}

            for new_box_identifier, priority_queue in ordered_distances.values():
                dist, box_identifier = priority_queue.get_nowait()

                vehicle_identifier = self.__previous_mapping[box_identifier]

                try:
                    if temp_inverse_mapping[vehicle_identifier][0] <= dist:
                        vehicle_identifier = self.__add_vehicle(
                            new_boxes[new_box_identifier])

                        temp_inverse_mapping[vehicle_identifier] = (
                            dist, box_identifier)

                        new_mapping[new_box_identifier] = vehicle_identifier

                except KeyError:
                    temp_inverse_mapping[vehicle_identifier] = (
                        dist, box_identifier)

                    new_mapping[new_box_identifier] = vehicle_identifier

        print(self.__previous_mapping)
        print(self.__vehicles)
        # TODO
        """