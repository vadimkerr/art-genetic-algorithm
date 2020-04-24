from typing import Tuple

import cv2
import numpy as np


def open(path: str) -> np.ndarray:
    return cv2.imread(path)


def save(image: np.ndarray, path: str) -> None:
    cv2.imwrite(filename=path, img=image)


def random(shape: Tuple) -> np.ndarray:
    return np.random.randint(low=0, high=255, size=shape)


def difference(image: np.ndarray, target_image: np.ndarray) -> int:
    return ((image - target_image) ** 2).sum()


def fitness(image: np.ndarray, target_image: np.ndarray) -> float:
    return 1 / difference(image, target_image)


def crossover(image_1: np.ndarray, image_2: np.ndarray) -> np.ndarray:
    height = image_1.shape[0]
    crossover_start = np.random.randint(low=0, high=height)

    return np.concatenate((image_1[:crossover_start], image_2[crossover_start:]))


def mutate(image: np.ndarray) -> np.ndarray:
    height = image.shape[0]

    mutation_start = np.random.randint(low=0, high=height)
    mutation_end = np.random.randint(low=mutation_start, high=height)

    random_mixin = random((mutation_end - mutation_start, *image.shape[1:]))
    return np.concatenate((image[:mutation_start], random_mixin, image[mutation_end:]))
