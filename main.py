# -*- coding: utf-8 -*- noqa
"""
Created on Fri Oct 18 13:09:35 2024

@author: Joel Tapia Salvador
"""
import cv2
import os
import re

import numpy as np


from time import monotonic_ns

from boxes_utils import centroids_distance
from detector import Detector
from general_utils import load_video, read_frame, show_image_on_window
from image_transformations import add_boxes, background_substraction


PATH_TO_DATA_DIRECTORY = os.path.join(".", "data")
PROCESSING_FPS = 5

ACCEPTED_FILE_FORMATS = (".mp4")
PATH_TO_MODEL = os.path.join('.', 'models', 'yolov8n.pt')

SHOW_RESULTS = False

separator = "\n" + "=" * os.get_terminal_size()[0]


def main():
    """
    Execute main logic of the program.

    Returns
    -------
    None.

    """
    file_regex_pattern = re.compile(r"(?:\.[a-zA-Z0-9]+$)")

    list_of_files = [os.listdir(PATH_TO_DATA_DIRECTORY)[1]]

    car_detector = Detector(PATH_TO_MODEL, SHOW_RESULTS)

    # frame = False

    last_state = None

    for file_name in list_of_files:
        file_format = file_regex_pattern.search(file_name).group()

        if file_format in ACCEPTED_FILE_FORMATS:
            print(separator)
            print(f'\n{file_name}')

            start_time = monotonic_ns()

            video = load_video(PATH_TO_DATA_DIRECTORY, file_name)

            duration_video = (video.get(cv2.CAP_PROP_FRAME_COUNT)
                              * 1000) / video.get(cv2.CAP_PROP_FPS)

            last_frame = read_frame(video, PROCESSING_FPS)

            frame = read_frame(video, PROCESSING_FPS)

            while frame is not None:

                movement_mask = background_substraction(frame, last_frame)

                bounding_boxes = car_detector.detect_cars(frame, SHOW_RESULTS)

                bounding_boxes.filter_boxes(movement_mask)

                coordinates_bounding_boxes = []

                current_state = []

                labels = []

                for box in bounding_boxes:
                    coordinates_bounding_boxes.append(box.xyxy)
                    current_state.append(box.centroid)
                    labels.append(f'{box.class_type}: {box.confiance:.2}')

                current_state = np.array(current_state)

                if SHOW_RESULTS:
                    show_image_on_window(
                        add_boxes(
                            frame,
                            coordinates_bounding_boxes,
                            labels,
                        )
                    )

                if (
                    isinstance(last_state, np.ndarray)
                ) and (
                    len(current_state) != 0
                ) and (
                    len(last_state) != 0
                ):
                    dist = centroids_distance(current_state, last_state)
                last_state = current_state
                last_frame = frame

                frame = read_frame(video, PROCESSING_FPS)

            end_time = monotonic_ns()

            time_lapsed = (end_time - start_time) // 1000000

            print(time_lapsed, duration_video)

    print(separator)


if __name__ == "__main__":
    main()
