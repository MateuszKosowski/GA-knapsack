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
    # CROSS_PROBABILITIES = [0.6, 0.8, 1.0]
    # MUTATION_PROBABILITIES = [0.01, 0.05, 0.1]
    # POPULATION_SIZES = [50, 100, 200]
    # ITERATIONS = [200, 500, 1000]
    # experiments = list(product(POPULATION_SIZES, CROSS_PROBABILITIES, MUTATION_PROBABILITIES, ITERATIONS))
    experiments = [
        # PM = 0.05, N = 100, T = 200
        (0.6, 0.05, 100, 200),
        (0.8, 0.05, 100, 200),
        (1.0, 0.05, 100, 200),
        # Pc = 0.8, N = 100, T = 200
        (0.8, 0.01, 100, 200),
        #(0.8, 0.05, 100, 200),
        (0.8, 0.1, 100, 200),
        # Pc = 0.08, Pm = 0.05, T = 200
        (0.8, 0.05, 50, 200),
        #(0.8, 0.05, 100, 200),
        (0.8, 0.05, 200, 200),
        #Pc = 0.8, Pm = 0.05, N = 100
        #(0.8, 0.05, 100, 200),
        (0.8, 0.05, 100, 500),
        (0.8, 0.05, 100, 1000),
    ]
    results = pd.DataFrame(columns=["Populacja", "Prawdopodobieństwo krzyżowania", "Prawdopodobieństwo mutacji", "Ilość iteracji", "Średnia", "Mediana", "Minimum", "Maksimum", "Odchylenie standardowe", "Czas wykonywania"])
    results_for_boxplot = pd.DataFrame(columns=["Populacja", "Prawdopodobieństwo krzyżowania", "Prawdopodobieństwo mutacji", "Ilość iteracji", "Maksimum"])
    print("\t".join(results.columns))

    for ex_idx, (cross, mut, pop, iterations) in enumerate(experiments):
        plt.clf()

        ex_results = Parallel(n_jobs=-1)(
            delayed(run_experiment)(pop, cross, mut, iterations) for _ in range(25)
        )    

        best_histories, times = zip(*ex_results)
        
        best_array = np.array([hist[-1] for hist in best_histories])
        avg_time = sum(times) / len(times)
        results.loc[len(results)] = {
            "Populacja": pop, 
            "Prawdopodobieństwo krzyżowania": cross, 
            "Prawdopodobieństwo mutacji": mut, 
            "Ilość iteracji": iterations,
            "Średnia": best_array.mean(), 
            "Mediana": np.median(best_array), 
            "Minimum": best_array.min(), 
            "Maksimum": best_array.max(), 
            "Odchylenie standardowe": round(best_array.std(), 2),
            "Czas wykonywania": round(avg_time, 2)
        }

        for run_hist, runtime in ex_results:
            results_for_boxplot.loc[len(results_for_boxplot)] = {
                "Populacja": pop,
                "Prawdopodobieństwo krzyżowania": cross,
                "Prawdopodobieństwo mutacji": mut,
                "Ilość iteracji": iterations,
                "Maksimum": run_hist[-1]
            }


        last_row = results.iloc[-1]
        print("\t".join(str(val) for val in last_row))

        for i, y_vals in enumerate(best_histories[:5], 1):
            plt.plot(range(iterations), y_vals, label=f"Próba {i}")

        plt.title(f"Populacja: {pop}, Krzyżowanie: {cross}, Mutacja: {mut}")
        plt.xlabel("Iteracje")
        plt.ylabel("Wartość")
        plt.legend(loc="lower right")
        plt.savefig(f"output/p{pop}_c{cross}_m{mut}.png")  

    
    results.to_csv("output/knapsack_results.csv", index=False, encoding="utf-8-sig")
    
    # === POPULATION SWEEP ===
    pop_sweep = results_for_boxplot[
        (results_for_boxplot["Prawdopodobieństwo krzyżowania"] == 0.8) &
        (results_for_boxplot["Prawdopodobieństwo mutacji"] == 0.05) &
        (results_for_boxplot["Ilość iteracji"] == 200)
    ]

    plt.clf()
    sns.boxplot(x="Populacja", y="Maksimum", data=pop_sweep)
    plt.title("Wpływ rozmiaru populacji na najlepsze rozwiązanie")
    plt.ylabel("Najlepszy wynik")
    plt.tight_layout()
    plt.savefig("output/pop_on_solution.png")

    # === CROSSOVER SWEEP ===
    cross_sweep = results_for_boxplot[
        (results_for_boxplot["Prawdopodobieństwo mutacji"] == 0.05) &
        (results_for_boxplot["Populacja"] == 100) &
        (results_for_boxplot["Ilość iteracji"] == 200)
    ]

    plt.clf()
    sns.boxplot(x="Prawdopodobieństwo krzyżowania", y="Maksimum", data=cross_sweep)
    plt.title("Wpływ prawdopodobieństwa krzyżowania na najlepsze rozwiązanie")
    plt.ylabel("Najlepszy wynik")
    plt.tight_layout()
    plt.savefig("output/cross_on_solution.png")

    # === MUTATION SWEEP ===
    mut_sweep = results_for_boxplot[
        (results_for_boxplot["Prawdopodobieństwo krzyżowania"] == 0.8) &
        (results_for_boxplot["Populacja"] == 100) &
        (results_for_boxplot["Ilość iteracji"] == 200)
    ]

    plt.clf()
    sns.boxplot(x="Prawdopodobieństwo mutacji", y="Maksimum", data=mut_sweep)
    plt.title("Wpływ prawdopodobieństwa mutacji na najlepsze rozwiązanie")
    plt.ylabel("Najlepszy wynik")
    plt.tight_layout()
    plt.savefig("output/mut_on_solution.png")

    # === ITERATION SWEEP ===
    iter_sweep = results_for_boxplot[
        (results_for_boxplot["Prawdopodobieństwo krzyżowania"] == 0.8) &
        (results_for_boxplot["Prawdopodobieństwo mutacji"] == 0.05) &
        (results_for_boxplot["Populacja"] == 100)
    ]

    plt.clf()
    sns.boxplot(x="Ilość iteracji", y="Maksimum", data=iter_sweep)
    plt.title("Wpływ liczby iteracji na najlepsze rozwiązanie")
    plt.ylabel("Najlepszy wynik")
    plt.xlabel("Iteracje")
    plt.tight_layout()
    plt.savefig("output/iter_on_solution.png")