# -*- coding: utf-8 -*- noqa
"""
Created on Fri Sep 27 12:50:37 2024

@author: Joel Tapia Salvador
"""

import os
import cv2
import numpy as np

from math import ceil


def create_path(path: str) -> None:
    """
    Create a directory.

    Recuservely creates all parents directory in the path if they don't exist.

    Parameters
    ----------
    path : string
        Path to the directory to be created.

    Returns
    -------
    None.

    """
    if os.path.isdir(path):
        return None

    root_path = os.path.split(path)[0]

    create_path(root_path)
    os.mkdir(path)


def load_video(path: str, name: str) -> cv2.VideoCapture:
    """
    Load a video into OpenCV video object.

    Parameters
    ----------
    path : str
        Path to the video file.
    name : str
        Name of the video file.

    Raises
    ------
    NotADirectoryError
        File path doesn't lead to a directory.
    FileNotFoundError
       File with given name not found in the given directory or is not a file.

    Returns
    -------
    OpenCV VideoCapture
        Video object containing the video frames and other information.

    """
    if not os.path.isdir(path):
        raise NotADirectoryError(f'"{path}" path is not a directory.')

    full_file_name = os.path.join(path, name)

    if not os.path.exists(full_file_name):
        raise FileNotFoundError(f'"{full_file_name}" path does not exist.')
    if not os.path.isfile(full_file_name):
        raise FileNotFoundError(f'"{full_file_name}" is not a file.')

    return cv2.VideoCapture(full_file_name)


def read_image(path: str, name: str) -> np.ndarray:
    """
    Read the image from a file using OpenCV.

    Parameters
    ----------
    path : string
        Path to the image file.
    name : string
        Name of the image file.

    Raises
    ------
    NotADirectoryError
        File path doesn't lead to a directory.
    FileNotFoundError
        File with given name not found in the given directory or is not a file.

    Returns
    -------
    numpy array
        Numpy array representing the image.

    """
    if not os.path.isdir(path):
        raise NotADirectoryError(f'"{path}" path is not a directory.')

    full_file_name = os.path.join(path, name)

    if not os.path.exists(full_file_name):
        raise FileNotFoundError(f'"{full_file_name}" path does not exist.')
    if not os.path.isfile(full_file_name):
        raise FileNotFoundError(f'"{full_file_name}" is not a file.')
    return cv2.imread(full_file_name)


# def read_frame(
#         video: cv2.VideoCapture,
#         fps: float | None
# ) -> np.ndarray | None:
def read_frame(video: cv2.VideoCapture, fps: float) -> np.ndarray:
    """
    Read the frames of a video.

    Reads at a given framerate slower or equal to the original video framerate.

    Parameters
    ----------
    video : cv2.VideoCapture
        VideoCaputer bject from OpenCV containing the video's frame to be read.
    fps : float or None, optional
        New framerate to read to. If new framerate is bigger than original
        video's fraemrate, original video's framerate will be used. If None
        given, original video's framerate used. The default is None.

    Returns
    -------
    frame : numpy array or None
        Frame represented as a Numpy array or None if there is no more frames
        left to read on the video.

    """
    if fps is None:
        fps = video.get(cv2.CAP_PROP_FPS)

    frame_selector = ceil(video.get(cv2.CAP_PROP_FPS) / fps)

    while (
            video.get(cv2.CAP_PROP_POS_FRAMES) % frame_selector != 0
    ) and (
        video.get(cv2.CAP_PROP_POS_FRAMES) < video.get(
            cv2.CAP_PROP_FRAME_COUNT)
    ):
        video.read()

    __, frame = video.read()
    
    if(frame is not None):
        frame = frame[600:]


    return frame


def remove_directory(path: str) -> None:
    """
    Remove a directory and all it's files.

    Recursively removes all child directories and their files too. To be able
    to delete the parent directory.

    Parameters
    ----------
    path : string
        String with the path to the directory to be removed.

    Returns
    -------
    None.

    """
    for child in os.listdir(path):
        child_path = os.path.join(path, child)
        if os.path.isdir(child_path):
            remove_directory(child_path)
        else:
            os.remove(child_path)
    os.rmdir(path)


def save_image(image: np.ndarray, path: str, name: str):
    """
    Save as a file a given image using OpenCV.

    Parameters
    ----------
    image : numpy array
        Numpy array representing the image.
    path : string
        Path to the image file.
    name : string
        Nome of the image file.

    Returns
    -------
    None.

    """
    create_path(path)

    cv2.imwrite(os.path.join(path, name), image)


def show_image_on_window(image: np.ndarray, window_name: str = "Image",
                         window_size: int = 1000) -> None:
    """
    Display the image on window.

    Parameters
    ----------
    image : numpy array
        Numpy array representing the image.
    window_name : string, optional
        Name the window will have. The default is "Image".
    window_size : integer, optional
        Max size of the window in pixels. The default is 1000 pixels.

    Returns
    -------
    None
        Returns none once the window is closed.

    """
    scale = max(image.shape[:2]) / window_size

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    cv2.resizeWindow(window_name, int(
        image.shape[1] // scale), int(image.shape[0] // scale))

    cv2.imshow(window_name, image)

    cv2.waitKey(0)

    cv2.destroyWindow(window_name)


if __name__ == "__main__":
    print(
        '\33[31m' + 'You are executing a module file, execute main instead.'
        + '\33[0m')
