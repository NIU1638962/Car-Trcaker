# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 00:32:37 2024

@author: joanb
"""

from boxes_utils import centroids_distance
import numpy as np

class Controller():
    
    def __init__(self):
        self.centroid = None
        self.control = False
        self.t = 0
        self.current_boxes = None
        
    def add_centroid(self, centroid):
        self.centroid = centroid 
        self.control = True
        self.t = 5
    
    def time(self):
        min_dist = 10000000
        if(self.centroid is not None):
            for box in self.current_boxes:
                dist  = centroids_distance(np.array([box.centroid]), np.array([self.centroid]))
                if(dist < min_dist): 
                    min_dist = dist
                    min_centroid = box.centroid
            print(min_dist)
        if(min_dist > 20):
            self.t-=1
        else:
            self.centroid = min_centroid
        if(self.t<1):
            self.t = 0
            self.centroid = None
            self.control = False
    