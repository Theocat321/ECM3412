"""
Small genetic algorithm for a toy bin-packing task.

I've used a class based approach to make the code super easy to read, modify, and experiment with.

Dataclasses are used for code readability! Learnt from the Django Module.

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
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class BPP:
    name: str
    k_items: int
    b_bins: int
    weights: List[float]


def make_bpp1() -> BPP:
    k = 500
    b = 10
    weights = [float(i) for i in range(1, k + 1)]
    return BPP("BPP1", k, b, weights)


def make_bpp2() -> BPP:
    k = 500
    b = 50
    weights = [ (i * i) / 2.0 for i in range(1, k + 1) ]
    return BPP("BPP2", k, b, weights)


@dataclass
class GAParams:
    population_size: int
    mutation_rate_per_gene: float
    tournament_size: int
    cross_over_rate: float = 0.8
    elitism: int = 1
    max_evals: int = 10000


@dataclass
class TrialResult:
    problem: str
    population_size: int
    mutation_rate_per_gene: float
    tournament_size: int
    cross_over_rate: float
    elitism: int
    max_evals: int
    trial_index: int
    seed: int
    best_fitness: float
    best_d: float
    generations: int
    evals_used: int


class BinPackingGA:
    def __init__(self, problem: BPP, params: GAParams, rng: random.Random):
        self.problem = problem
        self.params = params
        self.rng = rng
        self.k = problem.k_items
        self.b = problem.b_bins
        self.weights = problem.weights
        self.evals_used = 0

        self.history: Dict[str, List[float]] = {"gen": [], "best": [], "mean": []}

    def random_chromosome(self) -> List[int]:
        chrom: List[int] = []
        for _ in range(self.k):
            bin_id = self.rng.randint(1, self.b)
            chrom.append(bin_id)
        return chrom

    def _bins_sums(self, chrom: List[int]) -> Tuple[float, float]:
        sums = [0] * self.b
        for i, bin_id in enumerate(chrom):
            sums[bin_id - 1] += self.weights[i]
        return (min(sums), max(sums))

    def fitness(self, chrom: List[int]) -> float:
        min_sum, max_sum = self._bins_sums(chrom)
        d = max_sum - min_sum
        fit = 100.0 / (1.0 + d)
        self.evals_used += 1
        return fit

    def tournament_select(self, population: List[List[int]], fitnesses: List[float]) -> List[int]:
        t = self.params.tournament_size
        best_i = None
        for _ in range(t):
            i = self.rng.randrange(len(population))
            if best_i is None or fitnesses[i] > fitnesses[best_i]:
                best_i = i
        return population[best_i][:]

    def uniform_crossover(self, p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
        c1 = p1[:]
        c2 = p2[:]
        if self.rng.random() < self.params.cross_over_rate:
            for i in range(self.k):
                if self.rng.random() < 0.5:
                    c1[i], c2[i] = c2[i], c1[i]
        return c1, c2

    def mutate(self, chrom: List[int]) -> None:
        mutation_rate_per_gene = self.params.mutation_rate_per_gene
        for i in range(self.k):
            if self.rng.random() < mutation_rate_per_gene:
                chrom[i] = self.rng.randint(1, self.b)

    def run(self) -> Tuple[List[int], float, float, int, int]:
        p = self.params.population_size
        max_evals = self.params.max_evals
        n_elite = self.params.elitism

        pop = [self.random_chromosome() for _ in range(p)]
        fits = [self.fitness(ind) for ind in pop]

        best_idx = max(range(p), key=lambda i: fits[i])
        best = pop[best_idx][:]
        best_fit = fits[best_idx]
        mn, mx = self._bins_sums(best)
        best_d = mx - mn

        self.history["gen"].append(0)
        self.history["best"].append(max(fits))
        self.history["mean"].append(sum(fits) / len(fits))

        gens = 0

        while self.evals_used < max_evals:
            gens += 1

            elite_idx = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)[:n_elite]
            elites = [pop[i][:] for i in elite_idx]
            elite_fits = [fits[i] for i in elite_idx]

            new_pop: List[List[int]] = []
            new_fit: List[float] = []

            while len(new_pop) < p - n_elite and self.evals_used < max_evals:
                p1 = self.tournament_select(pop, fits)
                p2 = self.tournament_select(pop, fits)

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

            while len(new_pop) < p - n_elite:
                new_pop.append(elites[0][:])
                new_fit.append(elite_fits[0])

            pop = new_pop + elites
            fits = new_fit + elite_fits

            gbest_idx = max(range(p), key=lambda i: fits[i])
            if fits[gbest_idx] > best_fit:
                best = pop[gbest_idx][:]
                best_fit = fits[gbest_idx]
                mn, mx = self._bins_sums(best)
                best_d = mx - mn

            self.history["gen"].append(gens)
            self.history["best"].append(max(fits))
            self.history["mean"].append(sum(fits) / len(fits))

        return best, best_fit, best_d, gens, self.evals_used


def derived_seed(master_seed: int, problem_id: int, setting_id: int, trial_index: int) -> int:
    return master_seed + 10_000 * problem_id + 100 * setting_id + trial_index


def plot_history(history: Dict[str, List[float]], title: str, out_path: Path, show: bool = False) -> None:
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


def run_experiments(master_seed: int, csv_path: str, *, plot: bool = False, plot_dir: str = "plots", show: bool = False) -> List[TrialResult]:
    problems = [make_bpp1(), make_bpp2()]

    settings: List[GAParams] = [
        GAParams(population_size=100, mutation_rate_per_gene=0.01, tournament_size=3, cross_over_rate=0.8, elitism=1, max_evals=10_000),
        GAParams(population_size=100, mutation_rate_per_gene=0.05, tournament_size=3, cross_over_rate=0.8, elitism=1, max_evals=10_000),
        GAParams(population_size=100, mutation_rate_per_gene=0.01, tournament_size=7, cross_over_rate=0.8, elitism=1, max_evals=10_000),
        GAParams(population_size=100, mutation_rate_per_gene=0.05, tournament_size=7, cross_over_rate=0.8, elitism=1, max_evals=10_000),
    ]

    n_trials = 5
    results: List[TrialResult] = []

    best_by_cfg: Dict[Tuple[int, int], Tuple[float, Dict[str, List[float]], str, Path]] = {}

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

                if plot:
                    key = (pid, sid)
                    prev = best_by_cfg.get(key)
                    if (prev is None) or (best_fit > prev[0]):
                        fname = f"{problem.name.lower()}_set{sid}_best.png"
                        out_path = Path(plot_dir) / fname
                        title = f"{problem.name}, setting {sid} — best trial (seed {seed}, trial {tr})"
                        best_by_cfg[key] = (best_fit, ga.history.copy(), title, out_path)

    fieldnames = list(asdict(results[0]).keys()) if results else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    if plot:
        for (_pid, _sid), (_fit, hist, title, out_path) in best_by_cfg.items():
            plot_history(hist, title, out_path, show=show)

    return results


def summarize(results: List[TrialResult]) -> None:
    """
    Output to terminal the summary.
    """
    if not results:
        print("No results to summarize.")
        return

    from collections import defaultdict
    groups: Dict[Tuple[str, float, int], List[TrialResult]] = defaultdict(list)
    for r in results:
        key = (r.problem, r.mutation_rate_per_gene, r.tournament_size)
        groups[key].append(r)

    print("\n--- Summary ---")
    print("Problem, mutation_rate_per_gene, tournament, mean_best_fitness, std_best_fitness, mean_best_d, std_best_d")
    for key, rs in sorted(groups.items()):
        bf = [x.best_fitness for x in rs]
        bd = [x.best_d for x in rs]
        mean_bf = sum(bf) / len(bf)
        mean_bd = sum(bd) / len(bd)
        std_bf = (sum((x - mean_bf) ** 2 for x in bf) / (len(bf) - 1)) ** 0.5 if len(bf) > 1 else 0.0
        std_bd = (sum((x - mean_bd) ** 2 for x in bd) / (len(bd) - 1)) ** 0.5 if len(bd) > 1 else 0.0
        print(f"{key[0]}, {key[1]:.3f}, {key[2]}, {mean_bf:.8f}, {std_bf:.8f}, {mean_bd:.4f}, {std_bd:.4f}")


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
