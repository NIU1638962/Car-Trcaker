# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:13:53 2024

@author: JoelT
"""
import os
import json

from main import main, PATH_TO_DATA_DIRECTORY, RESULT_FILE, separator

GROUNDTRUTH_FILE = "metadata.json"


def test():
    with open(RESULT_FILE, "r") as file:
        result = json.load(file)

    with open(os.path.join(PATH_TO_DATA_DIRECTORY, GROUNDTRUTH_FILE), "r") as file:
        groundtruth = json.load(file)

    for file, metric in groundtruth.items():
        print(separator)
        print(file)
        try:
            metric_result = result[file]

            print(
                f'Down obtained: {metric_result["down"]}.'
                + f'Down expected: {metric["down"]}.'
                + f'({metric_result["down"] / metric["down"]:%})'
            )

            print(
                f'Up obtained: {metric_result["up"]}.'
                + f'Up expected: {metric["up"]}.'
                + f'({metric_result["up"] / metric["up"]:%})'
            )

        except KeyError:
            print("Error file wasn't processed in main.")

        print("")


if __name__ == "__main__":
    main()
    test()
