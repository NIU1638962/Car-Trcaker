# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 00:32:37 2024

@author: joanb
"""

class Controller():
    
    def __init__(self):
        self.centroid = None
        self.control = False
        self.t = 0
        
    def add_centroid(self, centroid):
        self.centroid = centroid 
        self.control = True
        self.t = 3
    
    def time(self):
        self.t-=1
        if(self.t<1):
            self.t = 0
            self.centroid = None
            self.control = False
    