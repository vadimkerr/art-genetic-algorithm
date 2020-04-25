import settings
from individual import Individual


class Population:
    def __init__(self, individuals=None):
        if individuals is None:
            individuals = [Individual() for _ in range(settings.POPULATION_SIZE)]

        self.individuals = sorted(
            individuals, key=lambda individual: individual.fitness
        )

    def evolve(self):
        new_individuals = self.individuals[settings.POPULATION_SIZE // 2 :]

        offsprings = []
        for i in range(1, settings.POPULATION_SIZE // 2):
            offspring = new_individuals[i - 1].crossover(new_individuals[i])
            offspring.mutate()
            offsprings.append(offspring)

        last_offspring = new_individuals[-1].crossover(new_individuals[0])
        last_offspring.mutate()
        offsprings.append(last_offspring)

        new_individuals += offsprings
        self.individuals = sorted(
            new_individuals, key=lambda individual: individual.fitness
        )

    @property
    def fitness(self):
        return self.individuals[-1].fitness

    def save_best(self, path):
        self.individuals[-1].save("best.png")


if __name__ == "__main__":
    population = Population()
    for i in range(settings.NUMBER_OF_ITERATIONS):
        population.evolve()
        print(f"{i+1}: {population.fitness}")

    population.save_best("best.png")
