# -*- coding: utf-8 -*- noqa
"""
Created on Fri Oct 18 13:09:35 2024

@author: Joel Tapia Salvador
"""
import cv2
import os
import re

import numpy as np

from math import ceil
from time import monotonic_ns

from boxes import Boxes
from boxes_utils import centroids_distance
from detector import Detector
from general_utils import load_video, read_frame, show_image_on_window
from image_transformations import add_boxes, background_substraction
from vehicles import Vehicles

from controller import Controller
import json

PATH_TO_DATA_DIRECTORY = os.path.join(".", "data")
RESULT_FILE = "result2.json"
PROCESSING_FPS = 5

ACCEPTED_FILE_FORMATS = (".mp4")
PATH_TO_MODEL = os.path.join('.', 'models', 'yolo11m.pt')

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

    list_of_files = os.listdir(PATH_TO_DATA_DIRECTORY)

    car_detector = Detector(PATH_TO_MODEL, SHOW_RESULTS)
    d = {}

    for file_name in list_of_files:
        file_format = file_regex_pattern.search(file_name).group()

        if file_format in ACCEPTED_FILE_FORMATS:
            print(separator)
            print(f'\n{file_name}')

            start_time = monotonic_ns()

            video = load_video(PATH_TO_DATA_DIRECTORY, file_name)

            video_duration = (video.get(cv2.CAP_PROP_FRAME_COUNT)
                              * 1000) / video.get(cv2.CAP_PROP_FPS)

            last_frame = read_frame(video, PROCESSING_FPS)

            frame = read_frame(video, PROCESSING_FPS)

            up = 0
            down = 0

            vehicles = Vehicles()
            cont = Controller()
            last_state = None

            while frame is not None:

                movement_mask = background_substraction(frame, last_frame)

                current_bounding_boxes = car_detector.detect_cars(
                    frame, SHOW_RESULTS)

                current_bounding_boxes.filter_boxes(movement_mask)

                coordinates_bounding_boxes = []

                current_state = []

                labels = []

                for box in current_bounding_boxes:
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
                """ 
                if (len(current_bounding_boxes) > 0):
                    show_image_on_window(add_boxes(
                        frame,
                        coordinates_bounding_boxes,
                        labels,
                    ), "frame")
                """
                vehicles.process(current_bounding_boxes, cont)

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
                """
                if (vehicles.up > up or vehicles.down > down):
                    up = vehicles.up
                    down = vehicles.down
                    print("up:", up, "    down:", down)
                    
                    show_image_on_window(last_frame, "Last_frame")

                    show_image_on_window(add_boxes(
                        frame,
                        coordinates_bounding_boxes,
                        labels,
                    ), "frame")
                """    
                frame = read_frame(video, PROCESSING_FPS)
                cont.time()

            end_time = monotonic_ns()

            time_elapsed = (end_time - start_time) / 1000000

            print(
                f'Time elapsed: {time_elapsed} '
                + f'({ceil(time_elapsed // (60 * 1000)):.0f} min)'
            )
            print(
                f'Duration video: {video_duration} '
                + f'({ceil(video_duration / (60 * 1000)):.0f} min)'
            )
            print(f'Real time: {(time_elapsed / video_duration) <= 1}')
            print(f'Vehicles Up: {vehicles.up}')
            print(f'Vehicles Down: {vehicles.down}')

            d[file_name] = {'down': vehicles.down, 'up': vehicles.up, 
                            'Real_time': (time_elapsed / video_duration) <= 1, 
                            'Time_elapsed': ceil(time_elapsed // (60 * 1000)), 
                            'Video_duration': ceil(video_duration / (60 * 1000))}
            with open(RESULT_FILE, 'w') as file:
                json.dump(d, file)
                
            print(separator)


if __name__ == "__main__":
    main()
