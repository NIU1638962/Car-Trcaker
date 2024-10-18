# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:09:35 2024

@author: Joel Tapia Salvador
"""
import cv2
import os
import re

from general_utils import load_video, read_frame, show_image_on_window


PATH_TO_DATA_DIRECTORY = os.path.join(".", "data")

ACCEPTED_FILE_FORMATS = (".mp4")
PROCESSING_FPS = 5


def main() -> None:
    """
    Executes main logic of the program.

    Returns
    -------
    None

    """
    file_regex_pattern = re.compile(r"(?:\.[a-zA-Z0-9]+$)")

    list_of_files = os.listdir(PATH_TO_DATA_DIRECTORY)

    for file_name in list_of_files:
        file_format = file_regex_pattern.search(file_name).group()

        if file_format in ACCEPTED_FILE_FORMATS:

            print(file_name)

            video = load_video(PATH_TO_DATA_DIRECTORY, file_name)

            frame = read_frame(video, PROCESSING_FPS)

            while frame is not None:
                frame = read_frame(video, PROCESSING_FPS)


if __name__ == "__main__":
    main()
