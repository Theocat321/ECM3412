"""
Small genetic algorithm for a toy bin-packing task.

I've used a class based approach to make the code super easy to read, modify, and experiment with.

Problems:
- BPP1: 500 items, 10 bins, weights w_i = i
- BPP2: 500 items, 50 bins, weights w_i = i^2 / 2

Chromosome:
- integer list of length k; each gene in [1..b] assigns item i -> bin gene[i].

Fitness:
- fitness = 100 / (1 + d), with d = (heaviest bin sum) - (lightest bin sum)

Usage:
  python main.py
  python main.py --master-seed 123 --csv ga_out.csv

TODO
- Completed!
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt


class BPP:
    def __init__(self, name, k_items, b_bins, weights):
        self.name = name
        self.k_items = k_items
        self.b_bins = b_bins
        self.weights = weights


def bpp1():
    k = 500
    b = 10
    weights = [float(i) for i in range(1, k + 1)]
    return BPP("BPP1", k, b, weights)


def bpp2():
    k = 500
    b = 50
    weights = [ (i * i) / 2.0 for i in range(1, k + 1) ]
    return BPP("BPP2", k, b, weights)


class GAParams:
    def __init__(
        self,
        population_size,
        mutation_rate_per_gene,
        tournament_size,
        cross_over_rate=0.8,
        elitism=1,
        max_evals=10000,
    ):
        self.population_size = population_size
        self.mutation_rate_per_gene = mutation_rate_per_gene
        self.tournament_size = tournament_size
        self.cross_over_rate = cross_over_rate
        self.elitism = elitism
        self.max_evals = max_evals


class TrialResult:
    def __init__(
        self,
        *,
        problem,
        population_size,
        mutation_rate_per_gene,
        tournament_size,
        cross_over_rate,
        elitism,
        max_evals,
        trial_index,
        seed,
        best_fitness,
        best_d,
        generations,
        evals_used,
    ):
        self.problem = problem
        self.population_size = population_size
        self.mutation_rate_per_gene = mutation_rate_per_gene
        self.tournament_size = tournament_size
        self.cross_over_rate = cross_over_rate
        self.elitism = elitism
        self.max_evals = max_evals
        self.trial_index = trial_index
        self.seed = seed
        self.best_fitness = best_fitness
        self.best_d = best_d
        self.generations = generations
        self.evals_used = evals_used

    def to_dict(self):
        # Preserve insertion order of attributes
        return {
            "problem": self.problem,
            "population_size": self.population_size,
            "mutation_rate_per_gene": self.mutation_rate_per_gene,
            "tournament_size": self.tournament_size,
            "cross_over_rate": self.cross_over_rate,
            "elitism": self.elitism,
            "max_evals": self.max_evals,
            "trial_index": self.trial_index,
            "seed": self.seed,
            "best_fitness": self.best_fitness,
            "best_d": self.best_d,
            "generations": self.generations,
            "evals_used": self.evals_used,
        }


class BinPackingGA:
    def __init__(self, problem, params, rng):
        self.problem = problem
        self.params = params
        self.rng = rng
        self.k = problem.k_items
        self.b = problem.b_bins
        self.weights = problem.weights
        self.evals_used = 0

        self.history = {"gen": [], "best": [], "mean": []}

    def _bins_diffs(self, chrom):
        sums = [0] * self.b
        for i, bin_id in enumerate(chrom):
            sums[bin_id - 1] += self.weights[i]
        return (min(sums), max(sums))

    def fitness(self, chrom):
        min_sum, max_sum = self._bins_diffs(chrom)
        d = max_sum - min_sum
        fit = 100.0 / (1.0 + d)
        self.evals_used += 1
        return fit

    def select_tournament(self, population, fitnesses):
        t = self.params.tournament_size
        best_i = None
        for _ in range(t):
            i = self.rng.randrange(len(population))
            if best_i is None or fitnesses[i] > fitnesses[best_i]:
                best_i = i
        return population[best_i][:]

    def uniform_crossover(self, p1, p2):
        c1 = p1[:]
        c2 = p2[:]
        if self.rng.random() < self.params.cross_over_rate:
            for i in range(self.k):
                if self.rng.random() < 0.5:
                    c1[i], c2[i] = c2[i], c1[i]
        return c1, c2

    def mutate(self, chrom):
        mutation_rate_per_gene = self.params.mutation_rate_per_gene
        for i in range(self.k):
            if self.rng.random() < mutation_rate_per_gene:
                chrom[i] = self.rng.randint(1, self.b)

    def random_chromosome(self):
        chrom = []
        for _ in range(self.k):
            bin_id = self.rng.randint(1, self.b)
            chrom.append(bin_id)
        return chrom

    def run(self):
        gens = 0

        p = self.params.population_size
        max_evals = self.params.max_evals
        n_elite = self.params.elitism

        pop = [self.random_chromosome() for _ in range(p)]
        fits = [self.fitness(ind) for ind in pop]

        best_idx = max(range(p), key=lambda i: fits[i])
        best = pop[best_idx][:]
        best_fit = fits[best_idx]
        mn, mx = self._bins_diffs(best)
        best_d = mx - mn

        self.history["gen"].append(0)
        self.history["best"].append(max(fits))
        self.history["mean"].append(sum(fits) / len(fits))

        while self.evals_used < max_evals:
            gens += 1

            indices = list(range(len(pop)))
            sorted_indices = sorted(indices, key=lambda i: fits[i], reverse=True)
            elite_idx = sorted_indices[:n_elite]

            # Copy the elite chromosomes from the population
            elites = []
            for i in elite_idx:
                elites.append(pop[i][:]) # create copy of each 

            # Store the fitness values of the elites
            elite_fits = []
            for i in elite_idx:
                elite_fits.append(fits[i])


            new_pop = []
            new_fit = []

            # Generate new individuals until we fill the population (excluding elites)
            while len(new_pop) < p - n_elite and self.evals_used < max_evals:
                p1 = self.select_tournament(pop, fits)
                p2 = self.select_tournament(pop, fits)

                c1, c2 = self.uniform_crossover(p1, p2)

                self.mutate(c1)
                self.mutate(c2)

                f1 = self.fitness(c1)
                if len(new_pop) < p - n_elite:
                    new_pop.append(c1)
                    new_fit.append(f1)

                if self.evals_used >= max_evals:
                    break

                f2 = self.fitness(c2)
                if len(new_pop) < p - n_elite:
                    new_pop.append(c2)
                    new_fit.append(f2)

            # If we have an odd number and one spot left, fill it with the best elite
            while len(new_pop) < p - n_elite:
                new_pop.append(elites[0][:])
                new_fit.append(elite_fits[0])

            # Create the new population
            pop = new_pop + elites
            fits = new_fit + elite_fits
            gbest_idx = max(range(p), key=lambda i: fits[i])

            # Update overall best if needed
            if fits[gbest_idx] > best_fit:
                best = pop[gbest_idx][:]
                best_fit = fits[gbest_idx]
                mn, mx = self._bins_diffs(best)
                best_d = mx - mn

            self.history["gen"].append(gens)
            self.history["best"].append(max(fits))
            self.history["mean"].append(sum(fits) / len(fits))

        return best, best_fit, best_d, gens, self.evals_used


def derived_seed(master_seed, problem_id, setting_id, trial_index):
    """
    Create a derived seed from the master seed used for each trial
    """
    return master_seed + 10_000 * problem_id + 100 * setting_id + trial_index


def plot_history(history, title, out_path, show=False):
    """
    Extra function used to plot for the report
    """
    gens = history.get("gen", [])
    best = history.get("best", [])
    mean = history.get("mean", [])
    if not gens:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(gens, best, label="best fitness", color="tab:blue")
    plt.plot(gens, mean, label="mean fitness", color="tab:orange", alpha=0.8)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out_path))
    if show:
        plt.show()
    plt.close()


def run_experiments(master_seed, csv_path, *, plot=False, plot_dir="plots", show=False):
    problems = [bpp1(), bpp2()]

    # Different GA parameter settings
    settings = [
        GAParams(population_size=100, mutation_rate_per_gene=0.01, tournament_size=3, cross_over_rate=0.8, elitism=1, max_evals=10_000),
        GAParams(population_size=100, mutation_rate_per_gene=0.05, tournament_size=3, cross_over_rate=0.8, elitism=1, max_evals=10_000),
        GAParams(population_size=100, mutation_rate_per_gene=0.01, tournament_size=7, cross_over_rate=0.8, elitism=1, max_evals=10_000),
        GAParams(population_size=100, mutation_rate_per_gene=0.05, tournament_size=7, cross_over_rate=0.8, elitism=1, max_evals=10_000),
    ]

    n_trials = 5
    results = []

    best_by_cfg = {}

    # Actual Experiment run
    for pid, problem in enumerate(problems):
        for sid, params in enumerate(settings):
            for tr in range(n_trials):
                seed = derived_seed(master_seed, pid, sid, tr)
                rng = random.Random(seed)
                ga = BinPackingGA(problem, params, rng)
                _, best_fit, best_d, gens, evals_used = ga.run()

                results.append(TrialResult(
                    problem=problem.name,
                    population_size=params.population_size,
                    mutation_rate_per_gene=params.mutation_rate_per_gene,
                    tournament_size=params.tournament_size,
                    cross_over_rate=params.cross_over_rate,
                    elitism=params.elitism,
                    max_evals=params.max_evals,
                    trial_index=tr,
                    seed=seed,
                    best_fitness=best_fit,
                    best_d=best_d,
                    generations=gens,
                    evals_used=evals_used,
                ))

                print(f"[{problem.name}] cfg={sid} tr={tr} seed={seed} "
                      f"best={best_fit:.8f} d={best_d:.4f} gens={gens} evals={evals_used}")

                # If we're graphinh, keep track
                if plot:
                    key = (pid, sid)
                    prev = best_by_cfg.get(key)
                    if (prev is None) or (best_fit > prev[0]):
                        fname = f"{problem.name.lower()}_set{sid}_best.png"
                        out_path = Path(plot_dir) / fname
                        title = f"{problem.name}, setting {sid} — best trial (seed {seed}, trial {tr})"
                        best_by_cfg[key] = (best_fit, ga.history.copy(), title, out_path)

    fieldnames = list(results[0].to_dict().keys()) if results else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())

    if plot:
        for (_pid, _sid), (_fit, hist, title, out_path) in best_by_cfg.items():
            plot_history(hist, title, out_path, show=show)

    return results


def summarize(results):
    """
    Output to terminal the summary.
    """
    if not results:
        print("No results to summarize.")
        return

    groups = defaultdict(list)
    for r in results:
        key = (r.problem, r.mutation_rate_per_gene, r.tournament_size)
        groups[key].append(r)

    print("\n--- Summary ---")
    print("Problem, mutation_rate_per_gene, tournament, mean_best_fitness, std_best_fitness, max_best_fitness, mean_best_d, std_best_d")
    for key, rs in sorted(groups.items()):
        bf = [x.best_fitness for x in rs]
        bd = [x.best_d for x in rs]
        mean_bf = sum(bf) / len(bf)
        mean_bd = sum(bd) / len(bd)
        max_bf = max(bf) if bf else 0.0
        std_bf = (sum((x - mean_bf) ** 2 for x in bf) / (len(bf) - 1)) ** 0.5 if len(bf) > 1 else 0.0
        std_bd = (sum((x - mean_bd) ** 2 for x in bd) / (len(bd) - 1)) ** 0.5 if len(bd) > 1 else 0.0
        print(f"{key[0]}, {key[1]:.3f}, {key[2]}, {mean_bf:.8f}, {std_bf:.8f}, {max_bf:.8f}, {mean_bd:.4f}, {std_bd:.4f}")


def main():
    parser = argparse.ArgumentParser(description="GA for BPPs")
    parser.add_argument("--master-seed", type=int, default=42, help="Master seed")
    parser.add_argument("--csv", type=str, default="ga_bpp_results.csv", help="Output CSV path")
    parser.add_argument("--plot", action="store_true", help="Plot best curves")
    parser.add_argument("--plot-dir", type=str, default="plots", help="Plot dir")
    parser.add_argument("--show-plots", action="store_true", help="Show plots")
    args = parser.parse_args()

    results = run_experiments(
        master_seed=args.master_seed,
        csv_path=args.csv,
        plot=args.plot,
        plot_dir=args.plot_dir,
        show=args.show_plots,
    )
    summarize(results)

    print(f"\nAll trial results written to: {args.csv}")


if __name__ == "__main__":
    main()
