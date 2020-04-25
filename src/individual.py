import cv2
import numpy as np

import settings


class Individual:
    def __init__(self, genes=None):
        if genes is None:
            genes = np.random.random(size=(settings.NUMBER_OF_CIRCLES, 5))
            genes[:, 0] *= settings.TARGET_IMAGE.shape[0]
            genes[:, 1] *= settings.TARGET_IMAGE.shape[1]
            genes[:, 2:] *= 255

        self.genes = genes.astype(np.uint)

    @property
    def image(self):
        image = np.zeros(settings.TARGET_IMAGE.shape)

        for gene in self.genes:
            image = cv2.circle(
                image,
                center=tuple(gene[:2]),
                radius=settings.CIRCLE_RADIUS,
                color=gene[2:].tolist(),
                thickness=cv2.FILLED,
            )

        return image

    def save(self, path):
        cv2.imwrite(path, self.image)

    def crossover(self, other):
        length = settings.NUMBER_OF_CIRCLES // 2
        new_genes = np.concatenate((self.genes[:length], other.genes[length:]))

        return Individual(new_genes)

    def mutate(self):
        num_genes_to_mutate = np.random.randint(
            0, settings.NUMBER_OF_CIRCLES * settings.MAX_MUT_RATIO
        )

        for i in range(num_genes_to_mutate):
            gene_idx = np.random.randint(0, settings.NUMBER_OF_CIRCLES)
            gene = self.genes[gene_idx]

            if np.random.random() <= 0.5:
                if np.random.random() <= settings.CENTER_MUT_PB:
                    gene[0] = np.random.randint(0, settings.TARGET_IMAGE.shape[0])
                    gene[1] = np.random.randint(0, settings.TARGET_IMAGE.shape[1])
            else:
                if np.random.random() <= settings.COLOR_MUT_PB:
                    gene[2:] = np.random.randint(0, 256, size=3)

    @property
    def fitness(self):
        return 1 / ((self.image - settings.TARGET_IMAGE) ** 2).sum()
