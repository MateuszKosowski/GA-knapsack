from ga_knapsack import Knapsack
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
import seaborn as sns
import time

def run_experiment(pop, cross, mut, iterations):
    start = time.perf_counter()
    knapsack = Knapsack(pop,cross,mut)
    knapsack.generate_initial_population()
    y_vals = []
    for i in range(iterations):
        knapsack.create_new_population()
        y_vals.append(knapsack.get_best_solution()[1])
    end = time.perf_counter()
    runtime = (end - start) * 1000
    return y_vals, runtime
    
if __name__ == "__main__":
    CROSS_PROBABILITIES = [0.6, 0.8, 1.0]
    MUTATION_PROBABILITIES = [0.01, 0.05, 0.1]
    POPULATION_SIZES = [50, 100, 200]
    ITERATIONS = 200
    experiments = list(product(POPULATION_SIZES, CROSS_PROBABILITIES, MUTATION_PROBABILITIES))
    results = pd.DataFrame(columns=["Populacja", "Prawdopodobieństwo krzyżowania", "Prawdopodobieństwo mutacji", "Średnia", "Mediana", "Minimum", "Maksimum", "Odchylenie standardowe", "Czas wykonywania"])

    print("\t".join(results.columns))

    for ex_idx, (pop, cross, mut) in enumerate(experiments):
        plt.clf()

        ex_results = Parallel(n_jobs=-1)(
            delayed(run_experiment)(pop, cross, mut, ITERATIONS) for _ in range(50)
        )    

        best_histories, times = zip(*ex_results)
        
        best_array = np.array([hist[-1] for hist in best_histories])
        avg_time = sum(times) / len(times)
        results.loc[len(results)] = {
            "Populacja": pop, 
            "Prawdopodobieństwo krzyżowania": cross, 
            "Prawdopodobieństwo mutacji": mut, 
            "Średnia": best_array.mean(), 
            "Mediana": np.median(best_array), 
            "Minimum": best_array.min(), 
            "Maksimum": best_array.max(), 
            "Odchylenie standardowe": round(best_array.std(), 2),
            "Czas wykonywania": round(avg_time, 2)
        }
        last_row = results.iloc[-1]
        print("\t".join(str(val) for val in last_row))

        for i, y_vals in enumerate(best_histories[:5], 1):
            plt.plot(range(ITERATIONS), y_vals, label=f"Próba {i}")

        plt.title(f"Populacja: {pop}, Krzyżowanie: {cross}, Mutacja: {mut}")
        plt.xlabel("Iteracje")
        plt.ylabel("Wartość")
        plt.legend(loc="lower right")
        plt.savefig(f"output/p{pop}_c{cross}_m{mut}.png")  

    
    results.to_csv("output/knapsack_results.csv", index=False, encoding="utf-8-sig")
    
    plt.clf()
    sns.boxplot(x="Populacja", y="Maksimum", data=results)
    plt.title("Efekty rozmiaru populacji na rozwiązanie")
    plt.savefig("output/pop_on_solution.png")

    plt.clf()
    sns.boxplot(x="Prawdopodobieństwo krzyżowania", y="Maksimum", data=results)
    plt.title("Efekty prawdopodobieństwa krzyżowania na rozwiązanie")
    plt.savefig("output/cross_on_solution.png")

    plt.clf()
    sns.boxplot(x="Prawdopodobieństwo mutacji", y="Maksimum", data=results)
    plt.title("Efekty prawdopodobieństwa mutacji na rozwiązanie")
    plt.savefig("output/mut_on_solution.png")


