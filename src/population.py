from functools import partial
from random import choice
from typing import List, Tuple

import numpy as np

import image
import settings


def random(shape: Tuple, length: int) -> List[np.ndarray]:
    return [image.random(shape) for _ in range(length)]


def pick_alfa(population: List[np.ndarray]) -> List[np.ndarray]:
    length = int(len(population) * settings.ALFA_RATIO)
    return population[-length:]


def pick_beta(population: List[np.ndarray]) -> List[np.ndarray]:
    length = int(len(population) * settings.BETA_RATIO)
    return population[-length:]


def sort_population(
    population: List[np.ndarray], target_image: np.ndarray
) -> List[np.ndarray]:
    return sorted(population, key=partial(image.fitness, target_image=target_image))


def evolve(population: List[np.ndarray], target_image: np.ndarray) -> List[np.ndarray]:
    new_population = pick_alfa(population)
    beta = pick_beta(population)
    for i in range(len(population) - len(new_population)):
        parent_1 = choice(beta)
        parent_2 = choice(beta)
        offspring = image.crossover(parent_1, parent_2)
        offspring = image.mutate(offspring)
        new_population.append(offspring)

    new_population = sort_population(new_population, target_image)
    return new_population


def fitness(population: List[np.ndarray], target_image: np.ndarray) -> float:
    return image.fitness(population[-1], target_image)


def save_best(population: List[np.ndarray], path: str) -> None:
    image.save(population[-1], path)


if __name__ == "__main__":
    target_image = image.open(settings.TARGET_IMAGE_PATH)

    population = random(target_image.shape, settings.POPULATION_SIZE)
    population = sort_population(population, target_image)

    for i in range(settings.NUMBER_OF_ITERATIONS):
        population = evolve(population, target_image)
        population_fitness = fitness(population, target_image)
        print(f"{i+1}: {population_fitness}")

    image.save(target_image, "target.png")
    save_best(population, "best.png")
