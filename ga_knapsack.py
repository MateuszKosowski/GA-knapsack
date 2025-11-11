import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

items_data = pd.read_csv('data/problem_plecakowy_dane_CSV_tabulatory.csv', sep='\t')

items_data.columns = ['index', 'name', 'weight', 'value']
items_data = items_data.iloc[:, 1:]
print(items_data.head())

for c in items_data.columns[1:]:
    items_data[c] = pd.to_numeric(items_data[c].str.replace(' ', ''), errors='coerce')

COUNT_OF_ITEMS = len(items_data)
BACKPACK_LOAD_CAPACITY = 6404180
COUNT_OF_ITERATIONS = 500


class Knapsack:
    def __init__(self, population_size, cross_probability, mutation_probability):
        self.population_size = population_size
        self.population = list()
        self.cross_probability = cross_probability
        self.mutation_probability = mutation_probability
        pass

    def generate_initial_population(self):
        pop = np.random.choice([True, False], size=(self.population_size, COUNT_OF_ITEMS))
        self.population = [self.calculate_fitness(chromosome) for chromosome in pop]

    def calculate_fitness(self, chromosome):
        weight = 0
        value = 0

        for i in range(len(chromosome)):
            genome = chromosome[i]
            if genome:
                weight += items_data.iloc[i].weight
                value += items_data.iloc[i].value

        if weight > BACKPACK_LOAD_CAPACITY:
            return chromosome, 0
        else:
            return chromosome, value

    def selection_rank(self):

        sorted_population = sorted(self.population, key=lambda item: item[1])
        sorted_chromosomes = [item[0] for item in sorted_population]
        rank_weights = [i + 1 for i in range(len(sorted_population))]

        selected = random.choices(
            population=sorted_chromosomes,
            weights=rank_weights,
            k=1
        )
        return selected[0]

    def single_cross(self, chromosome_1, chromosome_2):
        if np.array_equal(chromosome_1, chromosome_2):
            return chromosome_1.copy(), chromosome_2.copy()
        crossover_point = random.randint(1, COUNT_OF_ITEMS - 1)

        child_1 = np.concatenate([chromosome_1[:crossover_point], chromosome_2[crossover_point:]])
        child_2 = np.concatenate([chromosome_2[:crossover_point], chromosome_1[crossover_point:]])

        return child_1, child_2

    def mutate_bit_flip(self, chromosome):
        mutated_chromosome = chromosome.copy()
        for i in range(COUNT_OF_ITEMS):
            if random.random() < self.mutation_probability:
                mutated_chromosome[i] = not mutated_chromosome[i]

        return mutated_chromosome

    def create_new_population(self):
        new_population = []

        while len(new_population) < self.population_size:
            parent_1 = self.selection_rank()
            parent_2 = self.selection_rank()

            if random.random() < self.cross_probability:
                child_1, child_2 = self.single_cross(parent_1, parent_2)
            else:
                child_1, child_2 = parent_1.copy(), parent_2.copy()

            child_1 = self.mutate_bit_flip(child_1)
            child_2 = self.mutate_bit_flip(child_2)

            child_1 = self.calculate_fitness(child_1)
            child_2 = self.calculate_fitness(child_2)

            new_population.append(child_1)
            new_population.append(child_2)

        self.population = new_population

    def get_best_solution(self):
        best_solution = ([], 0)
        for solution in self.population:
            if best_solution[1] < solution[1]:
                best_solution = solution

        return best_solution


def main():
    knapsack = Knapsack(200, 0.9, 0.015)
    knapsack.generate_initial_population()
    y_vals = []
    for i in range(COUNT_OF_ITERATIONS):
        print(f"Iteration {i}/{COUNT_OF_ITERATIONS}")
        knapsack.create_new_population()
        print(f"Best solution: {knapsack.get_best_solution()[1]}")
        y_vals.append(knapsack.get_best_solution()[1])

    plt.plot(range(COUNT_OF_ITERATIONS), y_vals)
    plt.show()


if __name__ == '__main__':
    main()