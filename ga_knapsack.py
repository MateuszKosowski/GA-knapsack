import pandas as pd
import numpy as np
import random

items_data = pd.read_csv('data/problem_plecakowy_dane_CSV_tabulatory.csv', sep='\t')

items_data.columns = ['index', 'name', 'weight', 'value']
items_data = items_data.iloc[:, 1:]

for c in items_data.columns[1:]:
    items_data[c] = pd.to_numeric(items_data[c].str.replace(' ', ''), errors='coerce')

COUNT_OF_ITEMS = len(items_data)
BACKPACK_LOAD_CAPACITY = 6404180
COUNT_OF_ITERATIONS = 100
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
            return chromosome, 1
        else:
            return chromosome, value


    def roulette(self):
        # print(self.population)

        chromosomes = [item[0] for item in self.population]
        fitness_scores = [item[1] for item in self.population]

        selected = random.choices(
            population=chromosomes,
            weights=fitness_scores,
            k=1
        )

        print("selected: ")
        print(selected)
        print("\n")

        return selected[0]

    def signle_cross(self, chromosome_1, chromosome_2):
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
            parent_1 = self.roulette()
            parent_2 = self.roulette()

            # print("parent: \n")
            # print(parent_1)


            if random.random() < self.cross_probability:
                child_1, child_2 = self.signle_cross(parent_1, parent_2)
            else:
                child_1, child_2 = parent_1.copy(), parent_2.copy()
            # print(child_1)
            child_1 = self.mutate_bit_flip(child_1)
            child_2 = self.mutate_bit_flip(child_2)

            child_1 = self.calculate_fitness(child_1)
            child_2 = self.calculate_fitness(child_2)

            new_population.append(child_1)
            new_population.append(child_2)

        self.population = new_population


def main():
    knapsack = Knapsack(10, 0.9, 0.5)
    knapsack.generate_initial_population()
    # print(knapsack.population)

    for _ in range(COUNT_OF_ITERATIONS):
        knapsack.create_new_population()
        #print(knapsack.population)


if __name__ == '__main__':
    main()