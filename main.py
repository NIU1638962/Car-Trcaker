# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:09:35 2024

@author: Joel Tapia Salvador
"""
import cv2
import os
import re

from general_utils import load_video, read_frame, show_image_on_window, background_substraction, show_image_with_boxes
from boxes_utils import get_centroids, centroids_distance
from detector import Detector
import numpy as np


PATH_TO_DATA_DIRECTORY = os.path.join(".", "data")

ACCEPTED_FILE_FORMATS = (".mp4")
PROCESSING_FPS = 5

PATH_TO_MODEL = os.path.join('.', 'models', 'yolov8n.pt')


def main() -> None:
    """
    Executes main logic of the program.

    Returns
    -------
    None

    """
    file_regex_pattern = re.compile(r"(?:\.[a-zA-Z0-9]+$)")

    list_of_files = os.listdir(PATH_TO_DATA_DIRECTORY)

    det = Detector(PATH_TO_MODEL)

    frame = 1

    last_state = None
    last_frame = None

    for file_name in list_of_files:
        file_format = file_regex_pattern.search(file_name).group()

        if file_format in ACCEPTED_FILE_FORMATS:

            print(file_name)

            video = load_video(PATH_TO_DATA_DIRECTORY, file_name)
            last_frame = read_frame(video, PROCESSING_FPS)

            while frame is not None:
                frame = read_frame(video, PROCESSING_FPS)
                # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                movement_mask = background_substraction(frame, last_frame)

                res, bounding_boxes = det.detect_cars(frame)

                current_state, boxes = get_centroids(
                    bounding_boxes, movement_mask)

                show_image_with_boxes(frame.copy(), boxes)

                if (isinstance(last_state, np.ndarray)):
                    dist = centroids_distance(current_state, last_state)
                last_state = current_state
                last_frame = frame


if __name__ == "__main__":
    main()
