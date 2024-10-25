import numpy as np
from scipy.spatial.distance import cdist
import cv2


def get_centroids(boxes : np.ndarray, movement_mask):
    
    centers = []
    new_boxes = []
    
    for full_box in boxes:
        box = np.array(full_box.xyxy, dtype='int32').squeeze(0)
        
        
        segment = movement_mask[box[1]:box[3], box[0]:box[2]]
        
        
        movement = cv2.countNonZero(segment)
        area = segment.shape[0]*segment.shape[1]
        
        if(area > 0 and movement / area > 0.4):
            box_center = np.array([
                (box[2] + box[0]) / 2,
                (box[3] + box[1]) / 2])
            centers.append(box_center)
            new_boxes.append(full_box)
    return np.array(centers), new_boxes

def centroids_distance(centA, centB):
    return cdist(centA, centB)
