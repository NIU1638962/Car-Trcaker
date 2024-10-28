# -*- coding: utf-8 -*- noqa
"""
Created on Mon Sep 30 13:44:27 2024

@author: Joel Tapia Salvador
"""

import cv2
import numpy as np

from copy import deepcopy
from typing import List, Tuple


def absolute_difference(
        image_1: np.ndarray,
        image_2: np.ndarray
) -> np.ndarray:
    """
    Calculate the absolute difference between two images.

    Parameters
    ----------
    image_1 : numpy array
        Image represented as a numpy array.
    image_2 : numpy array
        Image represented as a numpy array.

    Returns
    -------
    numpy array
        Image of the absolute difference represented as numpy array.

    """
    return cv2.absdiff(image_1, image_2)


# def add_boxes(
#         image: np.ndarray,
#         boxes: List[Tuple[int, int, int, int]],
#         texts: List[str] | None = None,
# ) -> np.ndarray:
def add_boxes(
        image: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        texts: List[str] = None,
) -> np.ndarray:
    """
    Add boxes with labels to an image.

    Labels are optionals, if None passes will add no labels.

    Coordinates must be x top left corner, y top left corner,
    x bottom right corner, y bottom right corner.

    Parameters
    ----------
    image : numpy array
        Image to add the boxes to, represented as numpy array.
    boxes : List[Tuple[integer, integer, integer, integer]]
        List of coordinates xyxy of the boxes.
    texts : List[string] or None, optional
        List of strings with the labels of each box. The default is None.

    Returns
    -------
    image : numpy array
        Image with boxes added represented as numpy array.

    """
    image = deepcopy(image)

    if texts is None:
        texts = [None for i in range(len(boxes))]

    for box, text in zip(boxes, texts):
        x_min, y_min, x_max, y_max = box

        # Draw rectangle
        image = draw_rectangle(
            image,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),
            2
        )

        # Add text
        if text is not None:
            image = add_text(
                image,
                text,
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    return image


def add_text(
        image: np.ndarray,
        text: str,
        bottom_left_corner: Tuple[int, int, int],
        font,
        font_scale: float,
        colour: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 1
) -> np.ndarray:
    """
    Add text to the image.

    Parameters
    ----------
    image : numpy array
        Image to add text on, represented as a numpy array.
    text : string
        Text to add to the image.
    bottom_left_corner : Tuple[integer, integer, integer]
        Botom left corner of the text.
    font : OpenCV font
        Font of the text.
    font_scale : float
        Scale of the font.
    colour : Tuple[integer, integer, integer], optional
        Colour of the text to be added. The default is (255, 255, 255).
    thickness : int, optional
        Thickness of the lines of the text. The default is 1.

    Returns
    -------
    image : numpy array
        Image with the text added, represented as numpy array.

    """
    image = deepcopy(image)

    cv2.putText(
        image,
        text,
        bottom_left_corner,
        font,
        font_scale,
        colour,
        thickness
    )

    return image


def background_substraction(
        image: np.ndarray,
        background: np.ndarray
) -> np.ndarray:
    """
    Substract background from an image.

    Parameters
    ----------
    image : numpy array
        Image to substract background from, represented as numpy array.
    background : numpy array
        Background to substract, represented as numpy array.

    Returns
    -------
    dilated_movement_mask : numpy array
        Image with background substracted represented as numpy array.

    """
    image_gray = to_gray(image)
    background_gray = to_gray(background)

    difference = absolute_difference(image_gray, background_gray)

    movement_mask = binary_threshold(difference, 30, 255)

    dilated_movement_mask = dilate(movement_mask, None, iterations=2)

    return dilated_movement_mask


def binary_threshold(
        image: np.ndarray,
        threshold: int,
        max_value: int
) -> np.ndarray:
    """
    Apply a binary threshold to an image.

    Parameters
    ----------
    image : numpy array
        Image to get thresholded represented as numpy array.
    threshold : integer
        Number that the threshold is applied over.
    max_value : integer
        Value the threshold changes to.

    Returns
    -------
    result : numpy array
        Thresholded image represented as numpy array.

    """
    _, result = cv2.threshold(
        image,
        threshold,
        max_value,
        cv2.THRESH_BINARY
    )

    return result


def dilate(
        image: np.ndarray,
        kernel_size: Tuple[int, int],
        iterations: int = 1
) -> np.ndarray:
    """
    Dilate image.

    Parameters
    ----------
    image : numpy array
        Image to be dilated.
    kernel_size : Tuple[integet, integer]
        Tuple of 2 integer numebers with the size of the kernel.
    iterations : integer, optional
        Number of times dilation is applied. The default is 1.

    Returns
    -------
    numpy array
        Dilated image represented as numpy array.

    """
    return cv2.dilate(image, kernel_size, iterations=iterations)


def draw_circle(
        image: np.ndarray,
        center: Tuple[int, int],
        radius: int = 1,
        colour: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = -1
) -> np.ndarray:
    """


    Parameters
    ----------
    image : numpy array
        Image to draw de circle on, represented as numpy array.
    center : Tuple[int, int]
        Center of the circle.
    radius : integer, optional
        Radius of the circle. The default is 1.
    colour : Tuple[integer, integer, integer], optional
        Colour of the circle drawn. The default is (255, 255, 255).
    thickness : integer, optional
        Thicknes of the circle. -1 for filled. The default is -1.

    Returns
    -------
    image : numpy array
        Image with the circle drawn, represented as numpy array.

    """
    image = deepcopy(image)

    cv2.circle(image, center, radius, colour, thickness)

    return image


def draw_rectangle(
        image: np.ndarray,
        top_left_corner: Tuple[int, int],
        bottom_right_corner: Tuple[int, int],
        colour: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = -1,
) -> np.ndarray:
    """
    Draw a rectabgle on the image given.

    Parameters
    ----------
    image : numpy array
        Image to draw the rectangle on, represented as a numpy array.
    top_left_corner : Tuple[integer, integer]
        Top left corner where the rectangle will go.
    bottom_right_corner : Tuple[integer, integer]
        Bottom right corner where the rectangle will go.
    colour : Tuple[integer, integer, integer], optional
        Colour of the rectangle drawn. The default is (255, 255, 255).
    thickness : integer, optional
        Thickness of the rectangle borders. -1 for filled. The default is 1.

    Returns
    -------
    image : numpy array
        Image with the rectangle drawn, represented as numpy array.

    """
    image = deepcopy(image)

    cv2.rectangle(
        image,
        top_left_corner,
        bottom_right_corner,
        colour,
        thickness
    )

    return image


def gaussian_blur(
        image: np.ndarray,
        kernel_size: Tuple[int, int],
        mu: int = 0
) -> np.ndarray:
    """
    Blur an image applying Gaussian  Blur.

    Parameters
    ----------
    image : numpy array
        Image to be blured represented as numpy array.
    kernel_size : Tuple[interger, interger]
        Tuple of 2 integer numebers with the size of the kernel.
    mu : integer, optional
        Standard deviation of the Gaussian curve applied. The default is 0.

    Returns
    -------
    numpy array
        Blurred image represented as numpy array.

    """
    return cv2.GaussianBlur(image, kernel_size, mu)


def inverse(image: np.ndarray) -> np.ndarray:
    """
    Invert the colors of an image.

    Parameters
    ----------
    image : numpy array
        Image to be inverted represented as numpy array.

    Returns
    -------
    numpy array
        Inverse image represented as numpy array.

    """
    return 255 - image


def otsu_threshold(
        image: np.ndarray,
        threshold: int,
        max_value: int
) -> np.ndarray:
    """
    Apply an OTSU threshold to an image.

    Parameters
    ----------
    image : numpy array
        Image to get thresholded represented as numpy array.
    threshold : integer
        Number that the threshold is applied over.
    max_value : integer
        Value the threshold changes to.

    Returns
    -------
    result : numpy array
        Thresholded image represented as numpy array.

    """
    _, result = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return result


# def resize(
#         image: np.ndarray,
#         height: int | None = None,
#         width: int | None = None
# ) -> np.ndarray:
def resize(
        image: np.ndarray,
        height: int = None,
        width: int = None
) -> np.ndarray:
    """
    Resize image.

    Parameters
    ----------
    image : numpy array
        Image to resize represented as numpy array.
    height : integer or None, optional
        Height of the resulting image. If None keeps the original one.
        The default is None.
    width : integer or None, optional
        Width of the resulting image. If none keeps the original one.
        The default is None.

    Returns
    -------
    numpy array
        Resized image represented as a numpy array.

    """
    if height is None:
        height = image.shape[0]

    if width is None:
        width = image.shape[1]

    return cv2.resize(image, (height, width))


def to_gray(image: np.ndarray) -> np.ndarray:
    """
    Change the colour space of an image from RGB to Gray.

    Parameters
    ----------
    image : numpy array
        Image to get converted to Gray colour space represented as numpy array.

    Returns
    -------
    numpy array
        Image in Gray colour space represented as numpy array.

    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_hsv(image: np.ndarray) -> np.ndarray:
    """
    Change the colour space of an image from RGB to HSV.

    Parameters
    ----------
    image : numpy array
        Image to get converted to HSV colour space represented as numpy array.

    Returns
    -------
    numpy array
        Image in HSV colour space represented as numpy array.

    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


if __name__ == "__main__":
    print(
        '\33[31m' + 'You are executing a module file, execute main instead.'
        + '\33[0m')
