# -*- coding: utf-8 -*- noqa
"""
Created on Fri Oct 25 12:40:13 2024

@author: Joan Bernat
"""
import numpy as np

from scipy.spatial.distance import cdist


def centroids_distance(
        centroids_A: np.ndarray,
        centroids_B: np.ndarray
) -> np.ndarray:
    """
    Calculate the distance between all centroids of A and all centroids of B.

    Parameters
    ----------
    centA : numpy array
        Array of xy coordinates..
    centB : numpy array
        Array of xy coordinates..

    Returns
    -------
    numpy array
        Array of all possible pair differences.

    """
    return cdist(centroids_A, centroids_B)
